"""The UHF-SQD walker: sample, recover configurations, subsample per spin, solve with SBD.

Ported from kunalroy332/qcsc-prefect at 3968e58 (``algorithms/sbd/sbd/sqd.py``), UHF path only.
The recovery loop (``n_recovery_steps``), the K-batch scheme (``n_batches``), the per-spin
subsampling and carryover, the telemetry and the stored SQD arrays are the source's. Sampling goes
through a QuantumSource; the solver is ``qcsc_prefect_sbd.SBDSolverJob`` with unrestricted
integrals. Not present (deferred, see ``UHF_SQD_DEFERRED_FEATURES.md``): saved-pool loading,
checkpoints, shot batching, DD/twirling, the error-aware layout, the HCI and recombination
boosters, orbital optimization and the end-of-walker summary plot.
"""

import asyncio
import itertools
import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
from prefect import get_run_logger, task
from qcsc_prefect_sbd import SBDSolverJob
from qcsc_workflow_utility.chem import NpStrict1DArrayF64
from qcsc_workflow_utility.uhf import UHFElectronicProperties
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.configuration_recovery import (
    post_select_by_hamming_weight,
)
from qiskit_addon_sqd.configuration_recovery import (
    recover_configurations as _recover_configurations,
)
from qiskit_addon_sqd.counts import bit_array_to_arrays

from .data_io import save_ndarray
from .flow_params import CircuitParameters, SamplerParameters
from .lucj import create_lucj_circuit
from .np_type_extension import (
    NpStrict1DArrayLL,
    NpStrict2DArrayBool,
)
from .sampling import load_quantum_source

# Convert Addon function into Prefect Task
recover_configurations = task(_recover_configurations)

MODULE_RNG = np.random.default_rng(seed=1333)


def _spin_halves_as_ints(bitstrings: np.ndarray, norb: int) -> tuple[np.ndarray, np.ndarray]:
    """Split (beta-left, alpha-right) bool bitstrings into per-spin integer strings.

    Layout matches subsample_open_shell / recover_configurations: bits [:norb] are beta,
    bits [norb:] are alpha; bit 0 is the most-significant in the printed string.
    """
    weights = (1 << np.arange(norb - 1, -1, -1)).astype(np.int64)
    beta = (bitstrings[:, :norb].astype(np.int64) * weights).sum(axis=1)
    alpha = (bitstrings[:, norb:].astype(np.int64) * weights).sum(axis=1)
    return alpha, beta


def _compute_excitation_counts(ci_strings: np.ndarray, num_elec: int) -> dict[str, int]:
    """Compute excitation-level counts from Hartree-Fock.

    Returns dict with keys: HF, S, D, T, Q, high5 (>=5 excitations), max_exc, n_total, n_unique.
    """
    ci = np.asarray(ci_strings, dtype=np.int64).reshape(-1)
    if ci.size == 0:
        return {
            "HF": 0,
            "S": 0,
            "D": 0,
            "T": 0,
            "Q": 0,
            "high5": 0,
            "max_exc": 0,
            "n_total": 0,
            "n_unique": 0,
        }
    hf = (1 << num_elec) - 1
    exc = np.array([bin(int(x) ^ hf).count("1") // 2 for x in ci])
    return {
        "HF": int(np.sum(exc == 0)),
        "S": int(np.sum(exc == 1)),
        "D": int(np.sum(exc == 2)),
        "T": int(np.sum(exc == 3)),
        "Q": int(np.sum(exc == 4)),
        "high5": int(np.sum(exc >= 5)),
        "max_exc": int(exc.max()),
        "n_total": int(ci.size),
        "n_unique": len(set(ci.tolist())),
    }


def _excitation_summary(ci_strings: np.ndarray, num_elec: int) -> str:
    """Summarize a determinant list by excitation level from Hartree-Fock.

    The Hamiltonian only couples HF to single/double excitations (Slater-Condon), so a subspace
    dominated by high (>2) excitations cannot lower the energy below HF. This is the key diagnostic
    for why a large subspace can still collapse to the SCF energy. Returns a compact histogram.
    """
    counts = _compute_excitation_counts(ci_strings, num_elec)
    if counts["n_total"] == 0:
        return "empty"
    n_sd = counts["S"] + counts["D"]  # singles+doubles (the terms that couple to HF)
    n_high = counts["n_total"] - counts["HF"] - n_sd  # >2 (triples and above)
    return (
        f"n={counts['n_total']} unique={counts['n_unique']} HF={counts['HF']} "
        f"S={counts['S']} D={counts['D']} T={counts['T']} singles+doubles={n_sd} "
        f"higher(>2)={n_high} max_exc={counts['max_exc']}"
    )


def _reference_telemetry(elec_props: UHFElectronicProperties) -> dict:
    """Which Hamiltonian / reference a telemetry record belongs to. reference_* and
    fcidump_sha256 are None, as in the source for an FCIDUMP input (they are set only for a
    reference built from an active-space manifest, which is deferred)."""
    return {
        "norb": int(elec_props.num_orbitals),
        "nelec": [int(n) for n in elec_props.num_electrons],
        "reference_tag": getattr(elec_props, "reference_tag", None),
        "reference_source": getattr(elec_props, "reference_source", None),
        "fcidump_sha256": getattr(elec_props, "fcidump_sha256", None),
    }


# Fields of a persisted sample pool's reference record. The bitstrings live in the MO basis of
# the reference they were sampled from, so a pool is only valid for the same Hamiltonian
# (fcidump_sha256), sector (norb, nelec), reference state (reference_tag) and method.
_POOL_META_KEYS = ("norb", "nelec", "fcidump_sha256", "reference_tag", "method")

BIT_LAYOUT = "qiskit-bitarray-big-endian;meas[i]=q[i];jw:alpha=q[0:norb],beta=q[norb:2norb]"
"""Bit order of ``packed_bits`` in a sample pool.

``packed_bits`` is qiskit's big-endian packing of the ``meas`` register (clbit 0 is the least
significant bit of the last byte); ``create_lucj_circuit`` measures qubit i into clbit i; ffsim's
Jordan-Wigner layout puts the alpha orbitals on qubits 0..norb-1 and the beta orbitals on
norb..2norb-1. In ``bit_array_to_arrays`` form, columns [:norb] are beta and [norb:] alpha.
"""


def _pool_ref_meta(elec_props: UHFElectronicProperties) -> dict:
    """The reference record of a sample pool.

    The source's five keys (``_POOL_META_KEYS``), with ``fcidump_sha256`` filled with the sha256
    of the FCIDUMP the reference was built from (the source left it None for an FCIDUMP input),
    plus two keys the source's loader ignores:

    - ``reference_cache_key``: ``v2|<FCIDUMP sha256>|<guess token>|nelec=(n_alpha, n_beta)``. It
      identifies the BS-UHF reference by the FCIDUMP contents, the broken-symmetry guess and the
      electron numbers, and not by the code version. The bitstrings live in that reference's
      MO basis.
    - ``bit_layout``: :data:`BIT_LAYOUT`.
    """
    row = _reference_telemetry(elec_props)
    row["method"] = "uhf"
    row["fcidump_sha256"] = _fcidump_sha256(elec_props.reference_cache_key)
    meta = {k: row[k] for k in _POOL_META_KEYS}
    meta["reference_cache_key"] = elec_props.reference_cache_key
    meta["bit_layout"] = BIT_LAYOUT
    return meta


def _fcidump_sha256(reference_cache_key: str) -> str:
    """The FCIDUMP sha256 of a ``v2|<sha256>|<guess token>|nelec=...`` key.

    The guess token may itself contain ``|`` (it is ``"af=" + str(af_groups)``), so only the
    first two fields and the last one are split off.
    """
    parts = reference_cache_key.split("|", 2)
    if len(parts) != 3 or "|" not in parts[2]:
        raise ValueError(f"unexpected reference_cache_key: {reference_cache_key!r}")
    version, sha, rest = parts
    nelec = rest.rsplit("|", 1)[1]
    if version != "v2" or len(sha) != 64 or not nelec.startswith("nelec="):
        raise ValueError(f"unexpected reference_cache_key: {reference_cache_key!r}")
    return sha


def _pool_ref_meta_array(elec_props: UHFElectronicProperties) -> np.ndarray:
    """Reference record as a 1-element fixed-width unicode array (``allow_pickle=False`` npz
    cannot hold object arrays).

    The width is the source's ``<U2048``, or the JSON's length if that is longer: a fixed
    ``<U2048`` silently truncated a long record (for example with many AF-guess fragments).
    """
    text = json.dumps(_pool_ref_meta(elec_props), sort_keys=True)
    return np.array([text], dtype=f"<U{max(2048, len(text))}")


@task(
    task_run_name="run_sqd_#{trial_index:02d}-{walker_index}",
)
def walker_sqd(
    trial_index: int,
    walker_index: int,
    ucj_parameter: NpStrict1DArrayF64,
    circuit_params: CircuitParameters,
    elec_props: UHFElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    carryover: NpStrict2DArrayBool,
    sqd_dim: int,
    solver_block_name: str,
    quantum_source: str,
    random_seed: int,
    n_recovery_steps: int = 1,
    n_batches: int = 1,
    seed_cisd: int = 0,
    seed_budget_frac: float = 1.0,
    recomb_boost: bool = False,
    recomb_max_parents: int = 250,
    carryover_b: NpStrict2DArrayBool | None = None,
    sampler_params: SamplerParameters | None = None,
) -> tuple[
    tuple[float, tuple[NpStrict2DArrayBool, NpStrict2DArrayBool]],
    dict[str, Any],
]:
    """One DE walker: sample -> configuration recovery loop -> subsample -> SBD solve.

    Carryover is per spin: ``carryover`` holds the alpha determinants and ``carryover_b`` the
    beta determinants (a legacy stacked 2*norb ``carryover`` is also accepted). The returned
    carryover is the pair ``(alpha, beta)``.

    ``seed_cisd``, ``seed_budget_frac``, ``recomb_boost`` and ``recomb_max_parents`` keep the
    source's subsampling keywords at their defaults; the flow does not expose them.
    """
    logger = get_run_logger()
    davidson_solver = SBDSolverJob.load(solver_block_name)

    telemetry = {
        "trial_index": trial_index,
        "walker_index": walker_index,
        **_reference_telemetry(elec_props),
    }

    sampler_params = sampler_params or SamplerParameters()
    source = load_quantum_source(
        quantum_source,
        circuit_params=circuit_params,
        num_orbitals=elec_props.num_orbitals,
        random_seed=random_seed,
        trial_index=trial_index,
        walker_index=walker_index,
    )

    circuit = None
    if source.requires_circuit:
        circuit = create_lucj_circuit(
            ucj_parameter=ucj_parameter,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            n_lucj_layers=circuit_params.n_lucj_layers,
            use_reset_mitigation=circuit_params.use_reset_mitigation,
        )

    shots = int(sampler_params.shots)
    logger.info("Sampling from %s (shots=%s).", source.backend_name, shots)
    sampling_start = perf_counter()
    try:
        pub_result = source.sample(circuit, shots=shots)
    except Exception:
        logger.exception("Sampling failed on backend %s.", source.backend_name)
        raise
    logger.info(
        "Completed sampling on %s in %.2fs.",
        source.backend_name,
        perf_counter() - sampling_start,
    )

    if source.requires_circuit:
        # Measured on a device: reset mitigation and the sample pool, as in the source.
        meas_bits = pub_result[0].data.meas
        if circuit_params.use_reset_mitigation:
            test_bits = pub_result[0].data.test
            kept = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
            reset_fallback = len(kept) == 0
            if reset_fallback:
                # The reset-test register rejected every shot. This is expected on mock/simulator
                # backends that do not model the reset ancilla; on real hardware it would mean a
                # wholly failed submission. Either way it must not crash the flow
                # (BitArray.from_samples([]) raises), so fall back to the unmitigated shots and
                # warn.
                logger.warning(
                    "Reset mitigation kept 0/%d shots on %s; falling back to unmitigated shots.",
                    meas_bits.num_shots,
                    source.backend_name,
                )
                bit_array = meas_bits
            else:
                bit_array = BitArray.from_samples(kept, num_bits=meas_bits.num_bits)
        else:
            bit_array = meas_bits
        telemetry.update(
            shot_retention_rate=(
                float(bit_array.num_shots / meas_bits.num_shots) if meas_bits.num_shots else 0.0
            ),
        )
        if circuit_params.use_reset_mitigation:
            # Not in the source: shot_retention_rate is 1.0 both when every shot passed and when
            # every shot was rejected (fallback), so record which case it was.
            telemetry.update(
                reset_test_passed_shots=len(kept),
                reset_mitigation_fallback=reset_fallback,
            )

        # Persist the sample pool ONCE, before any diagonalization, so this (expensive,
        # real-hardware) sample can be re-diagonalized later without re-sampling. Best-effort:
        # a failed save must never abort a hardware run.
        try:
            pool_path = save_ndarray(
                file_prefix=f"raw_samples_t{trial_index:02d}_w{walker_index}",
                packed_bits=bit_array.array,
                num_bits=np.array([bit_array.num_bits], dtype=np.int64),
                num_shots=np.array([bit_array.num_shots], dtype=np.int64),
                ref_meta=_pool_ref_meta_array(elec_props),
            )
            telemetry["raw_samples_path"] = pool_path
            logger.info(
                "[persist] %d-shot pool (trial %d walker %d) saved: %s",
                int(bit_array.num_shots),
                trial_index,
                walker_index,
                pool_path,
            )
        except Exception:
            logger.exception("[persist] failed to save the sample pool (continuing).")
    else:
        bit_array = pub_result[0].data.meas

    logger.debug("Starting configuration recovery and diagonalization.")
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    norb = elec_props.num_orbitals
    num_elec_a, num_elec_b = elec_props.num_electrons

    # SQD self-consistency configuration-recovery loop.
    #
    # Canonical SQD iterates: recover configurations using the current average orbital
    # occupancies -> postselect -> subsample -> exactly diagonalize the subspace -> read updated
    # occupancies from the solver -> recover again. Recovery always re-uses the SAME raw quantum
    # samples (no re-sampling of the device); only the occupancies, and hence the recovered/
    # subsampled subspace, evolve. We keep the best (lowest-energy) pass.
    #
    # avg_occ starts from the UCCSD occupancies (per spin) and is refreshed
    # each pass from the SBD solver's orbital_occupancies. n_recovery_steps=1 reproduces the
    # previous single-pass behavior exactly.
    avg_occ = elec_props.initial_occupancy

    # Carry alpha / beta carryover as two independent (n, norb) arrays for the whole loop
    # (see _split_spin_carryover).
    carryover, carryover_b = _split_spin_carryover(
        carryover, norb, carryover_b, elec_props.num_electrons
    )

    # Per-walker random generator. recover_configurations and the subsample routines must draw
    # INDEPENDENT subspaces for each walker, otherwise differential evolution / multi-walker
    # averaging sees correlated draws and loses its diversity. A module-global Generator also races
    # under the threaded (concurrent) task runner. Derive a deterministic, per-(trial, walker)
    # seed -- reusing the same Cantor-pairing offset as the random-source path above -- so runs stay
    # reproducible while every walker is independent.
    walker_seed = int(
        random_seed
        + ((trial_index + walker_index) * (trial_index + walker_index + 1) // 2 + walker_index)
    )
    walker_rng = np.random.default_rng(seed=walker_seed)

    best_energy: float | None = None
    best_carryover: NpStrict2DArrayBool | None = None
    best_carryover_b: NpStrict2DArrayBool | None = None  # beta carryover of the best pass
    best_report_s3: str | None = None
    best_num_post = 0
    best_net_dim = 0

    # Per-recovery-step trajectory for offline analysis / presentation plots. telemetry otherwise
    # keeps only the single best pass, so the per-iteration curve (energy, subspace dim, spin
    # density, useful-det fraction, occupancies) would be unrecoverable except by log-scraping.
    # Purely diagnostic; appended once per pass and never read back inside the loop.
    recovery_trace: list[dict] = []
    # Previous step's carryover set (as a set of row-bytes) for the carryover-acquisition metric
    # |chi_co,g \ chi_co,g-1| -- the number of NEW carryover determinants introduced at step g
    # (arXiv:2405.05068 / the observability study arXiv:2512.05484 Fig. 4b). A decreasing trend
    # means the high-weight determinants are found early and then persist -> carryover working.
    prev_carryover_keys: set[tuple[str, bytes]] | None = None

    # One-time diagnostics on the raw quantum samples (before any recovery).
    logger.info(
        "[diag] raw samples: %d bitstrings, %d unique (norb=%d, nelec=(%d,%d))",
        raw_bitstrings.shape[0],
        len({row.tobytes() for row in raw_bitstrings}),
        norb,
        num_elec_a,
        num_elec_b,
    )

    for recovery_step in range(n_recovery_steps):
        # [diag] occupancies feeding this recovery pass (per-spin); fractional values are what
        # configuration recovery needs to bias toward correlated configs.
        occ_a_arr, occ_b_arr = np.asarray(avg_occ[0]), np.asarray(avg_occ[1])
        n_frac = int(np.sum((occ_a_arr > 0.01) & (occ_a_arr < 0.99))) + int(
            np.sum((occ_b_arr > 0.01) & (occ_b_arr < 0.99))
        )
        logger.info(
            "[diag] recovery %d/%d input occ: sum_a=%.3f sum_b=%.3f fractional=%d",
            recovery_step + 1,
            n_recovery_steps,
            float(occ_a_arr.sum()),
            float(occ_b_arr.sum()),
            n_frac,
        )

        bitstrings, probs = recover_configurations(
            bitstring_matrix=raw_bitstrings,
            probabilities=raw_probs,
            avg_occupancies=avg_occ,
            num_elec_a=num_elec_a,
            num_elec_b=num_elec_b,
            rand_seed=walker_rng,
        )
        bitstrings_post, probs_post = postselect_bitstrings(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            hamming_right=num_elec_a,
            hamming_left=num_elec_b,
        )
        # [diag] how many configs survive Hamming-weight post-selection (the usable subspace seed).
        logger.info(
            "[diag] recovery %d/%d post-select: %d / %d configs survived (%d unique)",
            recovery_step + 1,
            n_recovery_steps,
            bitstrings_post.shape[0],
            bitstrings.shape[0],
            len({row.tobytes() for row in bitstrings_post}) if bitstrings_post.shape[0] else 0,
        )

        # [diag] chemistry sanity: every post-selected config must sit in exactly the
        # (num_elec_a alpha, num_elec_b beta) sector, which fixes Sz = (na - nb)/2 exactly. The
        # bitstring layout is (beta-left [:norb], alpha-right [norb:]); see subsample_open_shell.
        if bitstrings_post.shape[0]:
            beta_pop = bitstrings_post[:, :norb].sum(axis=1)
            alpha_pop = bitstrings_post[:, norb:].sum(axis=1)
            sector_ok = bool(np.all(alpha_pop == num_elec_a) and np.all(beta_pop == num_elec_b))
            if not sector_ok:
                logger.warning(
                    "[diag] recovery %d/%d post-select LEAKED wrong (Na,Nb) sector: "
                    "alpha in %s, beta in %s (expected %d, %d)",
                    recovery_step + 1,
                    n_recovery_steps,
                    sorted(set(alpha_pop.tolist())),
                    sorted(set(beta_pop.tolist())),
                    num_elec_a,
                    num_elec_b,
                )

        if bitstrings_post.shape[0]:
            _alpha_i, _beta_i = _spin_halves_as_ints(bitstrings_post, norb)
            _hf_a = (1 << num_elec_a) - 1
            _hf_b = (1 << num_elec_b) - 1
            _tot = np.array(
                [
                    bin(int(a) ^ _hf_a).count("1") // 2 + bin(int(b) ^ _hf_b).count("1") // 2
                    for a, b in zip(_alpha_i, _beta_i)
                ]
            )
            _p = np.asarray(probs_post, dtype=np.float64)
            _psum = float(_p.sum())
            _p = _p / _psum if _psum > 0 else _p
            logger.info(
                "[exc-pop] recovery %d/%d prob-weighted population: "
                "HF=%.4f {1-2}=%.4f {3}=%.4f {4}=%.4f {>4}=%.4f",
                recovery_step + 1,
                n_recovery_steps,
                float(_p[_tot == 0].sum()),
                float(_p[(_tot >= 1) & (_tot <= 2)].sum()),
                float(_p[_tot == 3].sum()),
                float(_p[_tot == 4].sum()),
                float(_p[_tot > 4].sum()),
            )
            logger.info(
                "[exc-cnt] recovery %d/%d post-select config counts: "
                "HF=%d {1-2}=%d {3}=%d {4}=%d {>4}=%d  total=%d",
                recovery_step + 1,
                n_recovery_steps,
                int((_tot == 0).sum()),
                int(((_tot >= 1) & (_tot <= 2)).sum()),
                int((_tot == 3).sum()),
                int((_tot == 4).sum()),
                int((_tot > 4).sum()),
                int(_tot.size),
            )

        # K-batch SQD (arXiv:2405.05068): draw n_batches independent subspaces from the SAME
        # recovered distribution, diagonalize each, then take the MINIMUM energy as the variational
        # estimate and the MEAN occupancy across batches to feed the next recovery pass. A single
        # batch's occupancy is too noisy to steer recovery; averaging over batches stabilizes it.
        carryover_a = carryover  # per-spin arrays (normalized by _split_spin_carryover)

        batch_energy: float | None = None  # min over batches (the variational estimate)
        batch_carryover = None  # carryover of the best (lowest-energy) batch
        batch_carryover_b = None  # beta carryover of the best batch
        batch_net_dim = 0
        batch_save_kwargs: dict = {}
        batch_energies: list[float] = []  # every batch's total energy, for a min/mean/std summary
        occ_a_accum = np.zeros(norb, dtype=np.float64)
        occ_b_accum = np.zeros(norb, dtype=np.float64)

        for batch in range(n_batches):
            ci_a, ci_b = subsample_open_shell(
                bitstring_matrix=bitstrings_post,
                probabilities=probs_post,
                carryover_a=carryover_a,
                carryover_b=carryover_b,
                subspace_dim=sqd_dim,
                norb=norb,
                num_elec_a=num_elec_a,
                num_elec_b=num_elec_b,
                rng=walker_rng,
                seed_cisd=seed_cisd,
                seed_budget_frac=seed_budget_frac,
                recomb_boost=recomb_boost,
                recomb_max_parents=recomb_max_parents,
            )
            if batch == 0:
                # [diag] excitation profile of batch 0 (make-or-break: a subspace dominated by
                # >2 excitations cannot couple to HF and the energy stays at SCF).
                logger.info(
                    "[diag] recovery %d/%d alpha dets: %s",
                    recovery_step + 1,
                    n_recovery_steps,
                    _excitation_summary(ci_a, num_elec_a),
                )
                logger.info(
                    "[diag] recovery %d/%d beta  dets: %s",
                    recovery_step + 1,
                    n_recovery_steps,
                    _excitation_summary(ci_b, num_elec_b),
                )
            sbd_result = asyncio.run(
                davidson_solver.run(
                    ci_strings=(ci_a, ci_b),
                    one_body_tensor=elec_props.one_body_tensor,
                    two_body_tensor=elec_props.two_body_tensor,
                    norb=norb,
                    nelec=elec_props.num_electrons,
                    one_body_tensor_b=elec_props.one_body_tensor_b,
                    two_body_tensor_ab=elec_props.two_body_tensor_ab,
                    two_body_tensor_bb=elec_props.two_body_tensor_bb,
                )
            )
            this_carryover, this_carryover_b = _split_spin_carryover(
                sbd_result.carryover_bitstrings, norb, sbd_result.carryover_bitstrings_b
            )
            this_net_dim = int(len(ci_a) * len(ci_b))
            this_save_kwargs = dict(alphadets=ci_a, betadets=ci_b)

            this_energy = sbd_result.energy + elec_props.nuclear_repulsion_energy
            batch_energies.append(float(this_energy))
            # Accumulate occupancies (averaged over batches -> next recovery pass).
            occ_a_accum += np.asarray(sbd_result.orbital_occupancies[0], dtype=np.float64)
            occ_b_accum += np.asarray(sbd_result.orbital_occupancies[1], dtype=np.float64)
            logger.info(
                "[diag] recovery %d/%d batch %d/%d: energy = %.6f (net dim = %d)",
                recovery_step + 1,
                n_recovery_steps,
                batch + 1,
                n_batches,
                this_energy,
                this_net_dim,
            )
            # Keep the lowest-energy batch as this step's variational estimate.
            if batch_energy is None or this_energy < batch_energy:
                batch_energy = this_energy
                batch_carryover = this_carryover
                batch_carryover_b = this_carryover_b
                batch_net_dim = this_net_dim
                batch_save_kwargs = this_save_kwargs

        # [diag] report BOTH the reported (min) energy and the across-batch spread. The min is the
        # variational estimate we keep; the spread shows how lucky the best draw was and the K-batch
        # variance (large std => more batches / larger sqd_dim would likely help).
        if batch_energies:
            _be = np.asarray(batch_energies, dtype=np.float64)
            logger.info(
                "[diag] recovery %d/%d batch energies: min=%.6f mean=%.6f max=%.6f std=%.6f "
                "spread=%.1f mHa (n=%d)",
                recovery_step + 1,
                n_recovery_steps,
                float(_be.min()),
                float(_be.mean()),
                float(_be.max()),
                float(_be.std()),
                float((_be.max() - _be.min()) * 1000.0),
                _be.size,
            )

        energy = batch_energy
        step_carryover = batch_carryover
        step_carryover_b = batch_carryover_b
        net_dim = batch_net_dim
        save_kwargs = batch_save_kwargs
        # Mean occupancy across batches -> the occupancies that steer the NEXT recovery pass.
        step_occ_a = occ_a_accum / n_batches
        step_occ_b = occ_b_accum / n_batches

        # [diag] per-orbital spin density (n_alpha - n_beta) of the diagonalized state. Large
        # localized values indicate the spin polarization UHF is meant to capture; the sum equals
        # 2*Sz = (Na - Nb). This is the cheap, RDM-free spin check (full <S^2> needs the 2-RDM).
        spin_density = step_occ_a - step_occ_b
        logger.info(
            "[diag] recovery %d/%d spin density (n_a-n_b): sum=%.3f (=2Sz=%d) per-orb=%s",
            recovery_step + 1,
            n_recovery_steps,
            float(spin_density.sum()),
            num_elec_a - num_elec_b,
            np.array2string(spin_density, precision=3, max_line_width=200),
        )

        logger.info(
            "Recovery step %d/%d: best energy = %.6f over %d batch(es) (net subspace dim = %d)",
            recovery_step + 1,
            n_recovery_steps,
            energy,
            n_batches,
            net_dim,
        )

        report_s3 = save_ndarray(
            file_prefix="sqd_data",
            ucj_parameter=ucj_parameter,
            raw_bitstrings=raw_bitstrings,
            recovered_bitstrings=bitstrings,
            avg_occupancy=step_occ_a,
            avg_occupancy_b=step_occ_b,
            carryover=step_carryover,
            **({"carryover_b": step_carryover_b} if step_carryover_b is not None else {}),
            **save_kwargs,
        )

        # Keep the best (lowest-energy) pass.
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_carryover = step_carryover
            best_carryover_b = step_carryover_b
            best_report_s3 = report_s3
            best_num_post = len(bitstrings_post)
            best_net_dim = net_dim

        # Record this pass's trajectory point. net_dim / sqd_dim is the "useful determinant
        # fraction": how much of the requested subspace actually survived recovery + post-selection
        # into the diagonalized CI matrix (a subspace-quality proxy, see qcsc-bigdim memory). All
        # quantities are already computed above; this is a cheap dict append.
        _be = np.asarray(batch_energies, dtype=np.float64) if batch_energies else np.empty(0)

        # Carryover-acquisition metric |chi_co,g \ chi_co,g-1|: how many carryover determinants at
        # this step are NEW vs the previous step's carryover set. Each carryover row is a boolean
        # occupation vector; hash by (spin, raw bytes) to build sets. carryover_total = |chi_co,g|
        # (alpha + beta determinants, counted per spin).
        cur_keys = set()
        for _spin, _co in (("a", step_carryover), ("b", step_carryover_b)):
            if _co is not None and len(_co):
                cur_keys |= {(_spin, np.asarray(row, dtype=bool).tobytes()) for row in _co}
        carryover_total = len(cur_keys)
        if prev_carryover_keys is None:
            carryover_acquisition = carryover_total  # step 0: all carryover is "new"
        else:
            carryover_acquisition = len(cur_keys - prev_carryover_keys)
        prev_carryover_keys = cur_keys

        # Compute excitation counts (S/D/T/Q/higher) from the best batch's determinant lists,
        # separately for alpha and beta.
        exc_a = _compute_excitation_counts(save_kwargs.get("alphadets", np.array([])), num_elec_a)
        exc_b = _compute_excitation_counts(save_kwargs["betadets"], num_elec_b)

        recovery_trace.append(
            {
                "step": int(recovery_step),
                "energy": float(energy),
                "net_dim": int(net_dim),
                "n_post": int(len(bitstrings_post)),
                "sum_2Sz": float(spin_density.sum()),
                "useful_frac": float(net_dim / sqd_dim) if sqd_dim else 0.0,
                "batch_min": float(_be.min()) if _be.size else float(energy),
                "batch_mean": float(_be.mean()) if _be.size else float(energy),
                "batch_std": float(_be.std()) if _be.size else 0.0,
                "carryover_total": int(carryover_total),
                "carryover_acquisition": int(carryover_acquisition),
                "occ_a": step_occ_a.tolist(),
                "occ_b": step_occ_b.tolist(),
                # Excitation counts per spin (HF/S/D/T/Q/>=5) for per-step plotting.
                "exc_a": exc_a,
                "exc_b": exc_b,
            }
        )
        logger.info(
            "[diag] recovery %d/%d carryover: total=%d acquisition(new vs prev)=%d",
            recovery_step + 1,
            n_recovery_steps,
            carryover_total,
            carryover_acquisition,
        )

        # Feed the batch-averaged occupancies into the next recovery pass (self-consistency).
        avg_occ = (step_occ_a, step_occ_b)

        # Feed the best batch's carryover determinants forward too. Without this the loop re-uses
        # the ORIGINAL `carryover` argument every step (empty on a fresh walker), so with
        # carryover_type>0 the high-weight determinants found in step N are discarded at step N+1
        # and multi-step self-consistency loses the carryover half of its signal. The alpha / beta
        # arrays are carried separately (carryover, carryover_b).
        if step_carryover is not None:
            carryover = step_carryover
            carryover_b = (
                step_carryover_b
                if step_carryover_b is not None
                else np.empty((0, norb), dtype=bool)
            )

    logger.debug("Completed configuration recovery loop.")

    telemetry.update(
        num_post_determinants=best_num_post,
        net_subspace_dim=best_net_dim,
        energy=float(best_energy),
        n_recovery_steps=n_recovery_steps,
        n_batches=n_batches,
        # The source's telemetry keys, at their values in this scope.
        seed_cisd=seed_cisd,
        recomb_boost=recomb_boost,
        sqd_data=str(best_report_s3),
        recovery_trace=recovery_trace,
        last_updated=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return ((best_energy, (best_carryover, best_carryover_b)), telemetry)


def _split_spin_carryover(
    carryover: NpStrict2DArrayBool,
    norb: int,
    carryover_b: NpStrict2DArrayBool | None = None,
    num_electrons: tuple[int, int] | None = None,
) -> tuple[NpStrict2DArrayBool, NpStrict2DArrayBool]:
    """Normalize a UHF carryover input into separate ``(alpha, beta)`` (n, norb) bool arrays.

    Alpha and beta carryover are independent lists and generally differ in length (always when
    Na != Nb, and whenever the per-spin det lists differ). They are therefore carried as two arrays,
    never stacked row-wise: stacking forces the shorter spin to be zero-padded, and each all-zero
    padding row would be decoded as the empty determinant (popcount 0) and forced into the CI list.

    Accepted inputs:
      - ``carryover`` (n_a, norb) + ``carryover_b`` (n_b, norb): per-spin arrays (current format).
        ``carryover_b=None`` is accepted when ``carryover`` is a per-spin (n, norb) array: the beta
        list is then empty (the solver wrote no ``carryover_b.bin``).
      - ``carryover`` (n, 2*norb) with ``carryover_b=None``: legacy stacked (beta-left, alpha-right)
        layout from older checkpoints / OptimizerState; all-zero padding rows are dropped per half.
        A spin with 0 electrons (``num_electrons``) has the empty string as its only valid
        determinant, so its all-zero rows are kept (as a single row) instead of dropped.
      - an empty array: no carryover for either spin.
    """
    carryover = np.asarray(carryover, dtype=bool)
    no_beta = np.empty((0, norb), dtype=bool)
    if carryover_b is not None:
        co_a = carryover.reshape(-1, norb) if carryover.size else np.empty((0, norb), dtype=bool)
        co_b = np.asarray(carryover_b, dtype=bool)
        co_b = co_b.reshape(-1, norb) if co_b.size else no_beta
    elif carryover.size == 0:
        return np.empty((0, norb), dtype=bool), no_beta
    elif carryover.ndim == 2 and carryover.shape[1] == 2 * norb:
        co_b, co_a = carryover[:, :norb], carryover[:, norb:]
        nel_a, nel_b = num_electrons if num_electrons is not None else (None, None)
        co_b = _drop_padding_rows(co_b, nel_b)
        co_a = _drop_padding_rows(co_a, nel_a)
    elif carryover.ndim == 2 and carryover.shape[1] == norb:
        co_a, co_b = carryover, no_beta
    else:
        raise ValueError(
            f"UHF carryover must be per-spin (n, {norb}) arrays or a legacy stacked "
            f"(n, {2 * norb}) array; got shape {carryover.shape}."
        )
    return co_a, co_b


def _drop_padding_rows(rows: np.ndarray, num_elec: int | None) -> np.ndarray:
    """Drop all-zero (padding) rows, unless 0 electrons make the empty string the real
    determinant."""
    if num_elec == 0:
        return rows[:1] if len(rows) else rows
    return rows[rows.any(axis=1)]


@task
def postselect_bitstrings(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[NpStrict2DArrayBool, NpStrict1DArrayF64]:
    mask_postsel = post_select_by_hamming_weight(
        bitstring_matrix,
        hamming_right=hamming_right,
        hamming_left=hamming_left,
    )
    bs_mat_postsel = bitstring_matrix[mask_postsel]
    probs_postsel = probabilities[mask_postsel]
    probs_postsel = np.abs(probs_postsel) / np.sum(np.abs(probs_postsel))

    return bs_mat_postsel, probs_postsel


def _safe_run_logger():
    """Prefect run logger when inside a task/flow context, else None (e.g. unit tests)."""
    try:
        return get_run_logger()
    except Exception:
        return None


_CISD_CACHE: dict[tuple[int, int, int], NpStrict1DArrayLL] = {}


def _cisd_strings(norb: int, num_elec: int, level: int) -> NpStrict1DArrayLL:
    """Classically generate low-order excitation determinants of the Hartree-Fock reference.

    Returns the sorted-unique CI-string integers (EXCLUDING the HF string itself) for one spin
    channel, to be forced into the SQD subspace regardless of what the noisy quantum sampler
    produced. This implements the QSCI+SD / SCI-augmentation idea: augment the sampled configuration
    set with classically-generated single/double excitations so the low-excitation determinants that
    hardware noise drops or underweights are always present in the diagonalized subspace.

    Encoding matches the subsample routines, which build each CI string as
    ``sum_i occ[i] * 2**(norb-1-i)`` and set ``hartreefock = (1 << num_elec) - 1``. So the HF
    reference is exactly the integer with its lowest ``num_elec`` BITS set (bit positions
    ``0 .. num_elec-1``). We therefore work directly in BIT space: the "occupied" bit positions are
    ``0 .. num_elec-1`` and the "virtual" bit positions are ``num_elec .. norb-1``. Each excitation
    clears occupied bits and sets the same number of virtual bits, so the popcount stays exactly
    ``num_elec`` -> this spin's electron count is conserved, which (done per spin) preserves the
    (Na, Nb) sector and hence Sz exactly. (This bit-space view is equivalent to exciting electrons
    from occupied into virtual orbitals; it just avoids the orbital<->bit index flip.)

    Parameters
    ----------
    norb:
        Number of spatial orbitals for this spin channel.
    num_elec:
        Number of electrons in this spin channel (HF fills the lowest ``num_elec`` orbitals).
    level:
        Which excitations to generate. 0 -> none (empty array); 1 -> singles only;
        2 -> doubles only; 3 -> singles + doubles. Levels 1/2/3 mirror the ``seed_cisd`` flow
        parameter.

    Notes
    -----
    Counts (per spin): singles = ``num_elec * (norb - num_elec)``;
    doubles = ``C(num_elec, 2) * C(norb - num_elec, 2)``.
    Pure integer/bit operations (numpy + itertools) -> identical results on CPU and GPU hosts, no
    device code and no floating point. Results are cached per (norb, num_elec, level) because the
    set is constant across recovery steps and K-batches.

    References
    ----------
    Enhancing Accuracy of Quantum-Selected Configuration Interaction, J. Chem. Theory Comput.
    (PMC12423809); Molecular Quantum Computations on a Protein, arXiv:2512.17130; Auto-regressive
    Neural Quantum State Sampling for Selected Configuration Interaction, arXiv:2603.24728.
    """
    if level <= 0:
        return np.empty(0, dtype=np.longlong)
    key = (int(norb), int(num_elec), int(level))
    cached = _CISD_CACHE.get(key)
    if cached is not None:
        return cached

    hf = (1 << num_elec) - 1  # HF: lowest num_elec BITS set (bits 0 .. num_elec-1)
    occ = list(range(num_elec))  # occupied bit positions in HF
    virt = list(range(num_elec, norb))  # virtual (empty) bit positions in HF
    out: list[int] = []

    if level in (1, 3):  # singles: clear one occupied bit, set one virtual bit
        for i in occ:
            base = hf & ~(1 << i)
            for a in virt:
                out.append(base | (1 << a))
    if level in (2, 3):  # doubles: clear two occupied bits, set two virtual bits
        for i, j in itertools.combinations(occ, 2):
            base = hf & ~(1 << i) & ~(1 << j)
            for a, b in itertools.combinations(virt, 2):
                out.append(base | (1 << a) | (1 << b))

    arr = np.array(sorted(set(out)), dtype=np.longlong) if out else np.empty(0, dtype=np.longlong)
    arr = arr[arr != hf]  # HF is forced at index 0 by the caller; never duplicate it
    _CISD_CACHE[key] = arr
    return arr


def _merge_with_seed(
    hartreefock: int,
    ci_strs_carryover: NpStrict1DArrayLL,
    seed_strings: NpStrict1DArrayLL,
    subspace_dim: int,
    norb: int,
    num_elec: int,
    seed_cisd: int,
    new_strings_fn,
    logger=None,
    seed_budget_frac: float = 1.0,
) -> NpStrict1DArrayLL:
    """Assemble the final per-spin CI list ``[HF, carryover, seed, sampled]`` with seed capping.

    The forced determinants (HF + carryover + CISD seed) take priority within the subspace budget
    ``floor(sqrt(subspace_dim))``. If the CISD seed does not fit, it is capped so the run always
    proceeds at the requested ``subspace_dim``. ``new_strings_fn(budget)`` returns up to ``budget``
    sampled determinants (dedup vs HF/carryover/seed handled there); because the seed is excluded
    there, those sampled dets are exactly the ones NOT in the CISD manifold -- i.e. the
    higher-excitation configs the device sampled.

    ``seed_budget_frac`` (0 < f <= 1) caps the CISD seed to at most ``f * total_budget`` slots,
    reserving the rest of the subspace for sampled higher-excitation determinants. f=1.0 (default)
    forces as much seed as fits (original behavior); f<1.0 enables "partial-CISD + heavy mixing":
    e.g. f=0.5 fills half the budget with CISD S+D and the other half with the sample's best
    higher-excitation dets, which is what can beat the pure-CISD (S+D-only) energy ceiling.
    """
    seed_strings = np.asarray(seed_strings, dtype=np.longlong)
    # Dedup seed against carryover and HF so slots are not double-counted.
    if seed_strings.size:
        drop = np.isin(seed_strings, ci_strs_carryover) | (seed_strings == hartreefock)
        seed_strings = seed_strings[~drop]

    total_budget = (
        int(np.sqrt(subspace_dim)) - len(ci_strs_carryover) - 1
    )  # slots after HF+carryover
    total_budget = max(total_budget, 0)

    # Cap the seed to a fraction of the budget so sampled higher-excitation dets get the rest.
    frac = min(max(seed_budget_frac, 0.0), 1.0)
    seed_cap = int(total_budget * frac) if frac < 1.0 else total_budget
    n_seed_forced = min(len(seed_strings), seed_cap)
    seed_kept = seed_strings[:n_seed_forced]
    remaining = total_budget - n_seed_forced
    new_strings = new_strings_fn(remaining) if remaining > 0 else np.empty(0, dtype=np.longlong)

    if logger is not None and seed_cisd:
        logger.info(
            "[diag] seed CISD level=%d frac=%.2f: forced %d/%d seed dets/spin "
            "(budget=%d, sampled_higher=%d)",
            seed_cisd,
            frac,
            n_seed_forced,
            len(seed_strings),
            total_budget,
            len(new_strings),
        )
    return np.concatenate(
        ([hartreefock], ci_strs_carryover, seed_kept, new_strings), dtype=np.longlong
    )


@task
def subsample_open_shell(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    carryover_a: NpStrict2DArrayBool,
    carryover_b: NpStrict2DArrayBool,
    subspace_dim: int,
    norb: int,
    num_elec_a: int,
    num_elec_b: int,
    rng: np.random.Generator | None = None,
    seed_cisd: int = 0,
    seed_budget_frac: float = 1.0,
    recomb_boost: bool = False,
    recomb_max_parents: int = 250,
) -> tuple[NpStrict1DArrayLL, NpStrict1DArrayLL]:
    """Build separate alpha and beta CI string lists from bitstrings (open-shell / UHF).

    Unlike :func:`subsample_close_shell` (which averages and dedups across both spin halves of a
    single bitstring matrix and returns one alpha list), this keeps the alpha and beta pools fully
    independent: each spin gets its own deduplication, its own Hartree-Fock string at index 0, and
    its own carryover determinants. Returns the pair ``(ci_strs_a, ci_strs_b)`` consumed by the
    UHF SBD solver as distinct AlphaDets/BetaDets.

    The input bitstring layout matches the close-shell routine: ``bitstring_matrix[:, :norb]`` is
    the beta (left) half and ``bitstring_matrix[:, norb:]`` is the alpha (right) half.
    """
    # Assume longlong is 64 bit integer. Bit at index > 64 overflows.
    assert norb < 64

    num_configs = bitstring_matrix.shape[0]

    ci_strs_a = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_b = np.zeros(num_configs, dtype=np.longlong)
    for i in range(norb):
        ci_strs_b[:] += bitstring_matrix[:, i] * 2 ** (norb - 1 - i)
        ci_strs_a[:] += bitstring_matrix[:, norb + i] * 2 ** (norb - 1 - i)

    ci_a = _subsample_one_spin(
        ci_strs=ci_strs_a,
        probabilities=probabilities,
        carryover=carryover_a,
        subspace_dim=subspace_dim,
        norb=norb,
        num_elec=num_elec_a,
        rng=rng,
        seed_cisd=seed_cisd,
        seed_budget_frac=seed_budget_frac,
        recomb_boost=recomb_boost,
        recomb_max_parents=recomb_max_parents,
    )
    ci_b = _subsample_one_spin(
        ci_strs=ci_strs_b,
        probabilities=probabilities,
        carryover=carryover_b,
        subspace_dim=subspace_dim,
        norb=norb,
        num_elec=num_elec_b,
        rng=rng,
        seed_cisd=seed_cisd,
        seed_budget_frac=seed_budget_frac,
        recomb_boost=recomb_boost,
        recomb_max_parents=recomb_max_parents,
    )
    return ci_a, ci_b


def _subsample_one_spin(
    ci_strs: NpStrict1DArrayLL,
    probabilities: NpStrict1DArrayF64,
    carryover: NpStrict2DArrayBool,
    subspace_dim: int,
    norb: int,
    num_elec: int,
    rng: np.random.Generator | None = None,
    seed_cisd: int = 0,
    seed_budget_frac: float = 1.0,
    recomb_boost: bool = False,
    recomb_max_parents: int = 250,
) -> NpStrict1DArrayLL:
    """Dedup + carryover + HF-at-index-0 for a single spin channel.

    Mirrors the per-spin logic of :func:`subsample_close_shell` but operates on one spin's CI
    string integers and its own electron count and carryover, without averaging across spins.
    """
    if rng is None:
        rng = MODULE_RNG

    num_carryover = carryover.shape[0]

    # Hartree-Fock string for this spin, e.g. '0000011111' for num_elec=5.
    hartreefock = (1 << num_elec) - 1

    ci_strs_carryover = np.zeros(num_carryover, dtype=np.longlong)
    for i in range(norb):
        ci_strs_carryover[:] += carryover[:, i] * 2 ** (norb - 1 - i)

    ci_strs_unique, ci_probs_unique = _deduplicate_and_accumurate_probs(
        ci_strings=ci_strs,
        probabilities=probabilities,
    )

    # Remove HF string to make sure it appears at index 0.
    non_hf_mask = ci_strs_unique != hartreefock
    ci_strs_carryover = ci_strs_carryover[ci_strs_carryover != hartreefock]

    # Classically-seeded low-excitation determinants for this spin (QSCI+SD); empty if seed_cisd==0.
    seed_strings = _cisd_strings(norb, num_elec, seed_cisd)
    if recomb_boost:
        # The source's same-spin recombination booster (FE4S4_RECOMB_BOOST) is not ported yet.
        raise NotImplementedError("recomb_boost is not available in this port")

    def _draw_new(budget: int) -> NpStrict1DArrayLL:
        forced = (
            np.concatenate((ci_strs_carryover, seed_strings))
            if seed_strings.size
            else ci_strs_carryover
        )
        non_co_mask = ~np.isin(ci_strs_unique, forced)
        mask = non_hf_mask & non_co_mask
        cands = ci_strs_unique[mask]
        probs = ci_probs_unique[mask]
        if len(cands) > budget:
            return rng.choice(cands, size=budget, replace=False, p=probs / probs.sum())
        return cands

    # Carryover + CISD seed are always included; HF is forced to index 0.
    return _merge_with_seed(
        hartreefock=hartreefock,
        ci_strs_carryover=ci_strs_carryover,
        seed_strings=seed_strings,
        subspace_dim=subspace_dim,
        norb=norb,
        num_elec=num_elec,
        seed_cisd=seed_cisd,
        new_strings_fn=_draw_new,
        logger=_safe_run_logger(),
        seed_budget_frac=seed_budget_frac,
    )


def _deduplicate_and_accumurate_probs(
    ci_strings: NpStrict1DArrayLL,
    probabilities: NpStrict1DArrayF64,
) -> tuple[NpStrict1DArrayLL, NpStrict1DArrayF64]:
    ci_strs_unique, ci_strs_inv = np.unique(
        ci_strings,
        return_inverse=True,
    )
    ci_probs_unique = np.bincount(
        ci_strs_inv,
        weights=probabilities,
        minlength=len(ci_strs_unique),
    )
    return ci_strs_unique, ci_probs_unique
