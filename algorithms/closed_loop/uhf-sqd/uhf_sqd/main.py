"""The UHF-SQD closed loop: differential evolution over LUCJ parameters of a BS-UHF reference.

Ported from kunalroy332/qcsc-prefect at 3968e58 (``algorithms/sbd/sbd/main.py``, flow
``riken_sqd_de``), UHF path only. The DE trial, mutation, crossover and selection, and the
module RNG, are the source's. Orbital optimization between trials is deferred (see
``UHF_SQD_DEFERRED_FEATURES.md``). The task runner is an explicit ``ThreadPoolTaskRunner``; the
source's baseline ran on the same runner (``SBD_TASK_RUNNER=concurrent``), and Ray is not used.
"""

import os
import pathlib
from typing import Self

import numpy as np
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import RUN_ID, Inputs
from prefect.futures import PrefectFutureList
from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import BaseModel, Field
from qcsc_prefect_sbd import SBDSolverJob
from qcsc_workflow_utility import parse_block_ref
from qcsc_workflow_utility.chem import NpStrict1DArrayF64, NpStrict2DArrayF64
from qcsc_workflow_utility.uhf import UHFElectronicProperties, compute_uhf_integrals_from_fcidump

from .data_io import extend_table_artifact
from .flow_params import FlowParameters
from .lucj import initialize_ucj_parameters
from .np_type_extension import NpStrict2DArrayBool
from .sqd import walker_sqd

TELEMETRY_ARTIFACT_KEY = "uhf-sqd-telemetry"

MODULE_RNG = np.random.default_rng(seed=4574)


def _build_ab_indices(norb: int, stride: int) -> list[tuple[int, int]]:
    """Alpha-beta LUCJ coupling pairs (p, p), ordered by priority (most important first).

    On heavy-hex hardware the ab couplings are ancilla-mediated and can't all be realized, so
    ffsim's generate_lucj_pass_manager drops pairs from the END of the list until the layout
    fits. We therefore emit the stock stride-4 anchors FIRST (they always fit on heavy-hex),
    then the densification pairs (the extra p's a smaller stride adds) AFTER. This way a denser
    request degrades gracefully back toward the stock coupling instead of losing anchor pairs.
    stride=4 reproduces the historical [(p, p) for p in range(0, norb, 4)] exactly.
    """
    stride = max(1, int(stride))
    anchors = [(p, p) for p in range(0, norb, 4)]
    seen = {p for p, _ in anchors}
    extra = [(p, p) for p in range(0, norb, stride) if p not in seen]
    return anchors + extra


class OptimizerState(BaseModel):
    """Intermediate data for optimization."""

    energies: NpStrict1DArrayF64
    populations: NpStrict2DArrayF64
    carryover: NpStrict2DArrayBool
    # Beta carryover, kept separate from the alpha `carryover` (the two lists differ in length,
    # always for Na != Nb). None until the first carryover is selected.
    carryover_b: NpStrict2DArrayBool | None = None
    best_index: int | None = Field(
        default=None,
        ge=0,
    )

    def best_energy(self) -> float | None:
        if self.best_index is None:
            return None
        return float(self.energies[self.best_index])

    def copy(self) -> Self:
        return OptimizerState(
            energies=self.energies.copy(),
            populations=self.populations.copy(),
            carryover=self.carryover.copy(),
            carryover_b=None if self.carryover_b is None else self.carryover_b.copy(),
            best_index=self.best_index,
        )

    @classmethod
    def from_parameters(
        cls,
        num_walkers: int,
        norb: int,
        n_aa_params: int,
        n_ab_params: int,
        n_reps: int,
    ) -> "OptimizerState":
        num_lucj_params = n_reps * (n_aa_params + n_ab_params + norb**2) + norb**2
        return OptimizerState(
            energies=np.zeros(num_walkers, dtype=np.float64),
            populations=np.full((num_walkers, num_lucj_params), np.nan, dtype=np.float64),
            carryover=np.full((0, norb), np.nan, dtype=bool),
        )


@flow(
    name="uhf_sqd_de",
    task_runner=ThreadPoolTaskRunner(),
)
def uhf_sqd_de(
    parameters: FlowParameters,
):
    logger = get_run_logger()

    # Fail fast: the solver block must exist and read unrestricted integrals.
    slug, name = parse_block_ref(parameters.solver_block_ref)
    if slug != "sbd_solver_job":
        raise ValueError(
            f"solver_block_ref must be 'sbd_solver_job/<name>'. got: {parameters.solver_block_ref}"
        )

    try:
        solver = SBDSolverJob.load(name)
    except Exception:
        logger.exception("Failed to load solver block: %s", parameters.solver_block_ref)
        raise
    if solver.integrals != "unrestricted":
        raise ValueError(
            f"Solver block {parameters.solver_block_ref} has integrals={solver.integrals!r}; "
            "UHF-SQD needs integrals='unrestricted' (a -D_UHF SBD driver)."
        )

    logger.info(
        "Solver OK: ref=%s mode=%s integrals=%s",
        parameters.solver_block_ref,
        solver.solver_mode,
        solver.integrals,
    )

    telemetry_data = []
    create_table_artifact(
        table=telemetry_data,
        key=TELEMETRY_ARTIFACT_KEY,
        description="UHF-SQD intermediate data.",
    )

    # spin_sq is the solver's target, which SBD does not take; the reference <S^2> is a field
    # of the result (reference_spin_square).
    reference = parameters.reference
    elec_props = compute_uhf_integrals_from_fcidump(
        fcidump_file=parameters.fcidump,
        spin_sq=None,
        af_groups=reference.af_groups,
        cache_dir=reference.cache_dir,
        cache_read=reference.cache_read,
        ccsd_max_cycle=reference.ccsd_max_cycle,
        ccsd_restarts=reference.ccsd_restarts,
    )
    logger.info(
        "Reference: E_UHF=%.10f <S^2>=%.4f norb=%d nelec=%s key=%s",
        elec_props.reference_energy,
        elec_props.reference_spin_square,
        elec_props.num_orbitals,
        tuple(elec_props.num_electrons),
        elec_props.reference_cache_key,
    )

    # We assume heavy-hex topology
    # Orbitals for different spins have connections between every 4th orbital.
    aa_indices = [(p, p + 1) for p in range(elec_props.num_orbitals - 1)]
    ab_indices = _build_ab_indices(elec_props.num_orbitals, parameters.circ_params.ab_stride)
    logger.info(
        "LUCJ alpha-beta coupling: stride=%s -> %s pairs (stock stride-4 would give %s)",
        parameters.circ_params.ab_stride,
        len(ab_indices),
        len(range(0, elec_props.num_orbitals, 4)),
    )

    state = OptimizerState.from_parameters(
        num_walkers=parameters.de_params.num_walkers,
        norb=elec_props.num_orbitals,
        n_aa_params=len(aa_indices),
        n_ab_params=len(ab_indices),
        n_reps=parameters.circ_params.n_lucj_layers,
    )

    # Start differential evoluation
    for i in range(parameters.de_params.iterations):
        logger.info(f"Running differential evolution trial {i}")

        state = differential_evolution_trial(
            trial_index=i,
            parameters=parameters,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            state=state,
        )

        logger.info(f"Current best energy = {state.best_energy()} (walker {state.best_index})")

    return state.best_energy()


@task(
    task_run_name="de_trial#{trial_index:02d}",
    # Cache on the flow run ID and trial_index.
    # This is roughly identical with the conventional checkpoint mechanism.
    cache_policy=Inputs(
        exclude=[
            "parameters",
            "elec_props",
            "aa_indices",
            "ab_indices",
            "state",
        ]
    )
    + RUN_ID,
)
def differential_evolution_trial(
    trial_index: int,
    parameters: FlowParameters,
    elec_props: UHFElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    state: OptimizerState,
) -> OptimizerState:
    """Run one DE trial and return the new state."""
    logger = get_run_logger()

    if state.best_index is not None:
        if parameters.de_params.num_walkers < 4:
            # No differential evolution: DE mutation (a - b + c - d) needs >= 4 distinct
            # walkers. With fewer, re-evaluate the SAME population each trial.
            trial_populations = state.populations
        else:
            # Create next generation
            trial_populations = mutation_and_crossover(
                current_populations=state.populations,
                best_index=state.best_index,
                scaling_factor=parameters.de_params.fxc,
                crossover_rate=parameters.de_params.cr_prob,
            )
    else:
        # Initialize populations
        trial_populations = initialize_ucj_parameters(
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            num_walkers=parameters.de_params.num_walkers,
            randomization_factor=parameters.de_params.randomization_factor,
            n_lucj_layers=parameters.circ_params.n_lucj_layers,
            ucj_optimize=parameters.circ_params.ucj_optimize,
        )

    _, solver_block_name = parse_block_ref(parameters.solver_block_ref)

    futs = PrefectFutureList()
    for walker_index, ucj_parameter in enumerate(trial_populations):
        prefect_fut = walker_sqd.submit(
            trial_index=trial_index,
            walker_index=walker_index,
            ucj_parameter=ucj_parameter,
            circuit_params=parameters.circ_params,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            carryover=state.carryover,
            carryover_b=state.carryover_b,
            sqd_dim=parameters.sqd_dim,
            solver_block_name=solver_block_name,
            quantum_source=parameters.quantum_source,
            random_seed=parameters.random_seed,
            n_recovery_steps=parameters.n_recovery_steps,
            n_batches=parameters.n_batches,
            sampler_params=parameters.sampler,
        )
        futs.append(prefect_fut)

    # Collect results
    result_energies = np.full(parameters.de_params.num_walkers, np.nan, dtype=np.float64)
    # Per-walker (alpha, beta) carryover pairs.
    result_carryovers: list[tuple] = [None] * parameters.de_params.num_walkers
    records: list[dict] = [None] * parameters.de_params.num_walkers
    for walker_index, ((energy, carryover), telemery) in enumerate(futs.result()):
        result_energies[walker_index] = energy
        result_carryovers[walker_index] = carryover
        records[walker_index] = telemery

    # Update artifact
    artifact_id = extend_table_artifact(
        artifact_key=TELEMETRY_ARTIFACT_KEY,
        new_table=records,
    )
    logger.debug(f"Updated {TELEMETRY_ARTIFACT_KEY} artifact {str(artifact_id)}")

    new_state = selection(
        trial_populations=trial_populations,
        trial_energies=result_energies,
        trial_carryovers=result_carryovers,
        current_state=state,
    )
    return new_state


@task
def mutation_and_crossover(
    current_populations: NpStrict2DArrayF64,
    best_index: int,
    scaling_factor: float,
    crossover_rate: float,
) -> NpStrict2DArrayF64:
    global MODULE_RNG
    num_walkers, num_params = current_populations.shape

    if num_walkers < 4:
        # Each mutant draws 4 distinct other walkers (a - b + c - d); this is undefined below 4.
        # num_walkers < 4 is only allowed for a single evaluation pass (iterations = 1), which
        # never reaches this function, so reaching here with < 4 is a misconfiguration.
        raise ValueError(
            "Differential-evolution mutation requires num_walkers >= 4 "
            f"(got {num_walkers}); use iterations = 1 for a single-walker evaluation pass."
        )

    mutant = np.zeros_like(current_populations, dtype=np.float64)
    for i in range(num_walkers):
        r1, r2, r3, r4 = MODULE_RNG.choice(
            num_walkers,
            size=4,
            replace=False,
            shuffle=True,
        )
        drift_vec = (
            current_populations[r1]
            - current_populations[r2]
            + current_populations[r3]
            - current_populations[r4]
        )
        mutant[i] = current_populations[best_index] + scaling_factor * drift_vec

    for i in range(num_walkers):
        crossover_weights = MODULE_RNG.random(num_params)
        mask = crossover_weights > crossover_rate
        # Mutate at least one dimension
        index_to_keep = MODULE_RNG.choice(num_params, size=1)
        mask[index_to_keep] = False
        mutant[i, mask] = current_populations[i, mask]

    return mutant


@task
def selection(
    trial_populations: list[NpStrict2DArrayF64],
    trial_energies: NpStrict1DArrayF64,
    trial_carryovers: list[tuple[NpStrict2DArrayBool, NpStrict2DArrayBool | None]],
    current_state: OptimizerState,
) -> OptimizerState:
    logger = get_run_logger()

    new_state = current_state.copy()

    # The population array is pre-allocated from the spin-balanced parameter count in
    # OptimizerState.from_parameters. The spin-unbalanced (UHF) UCJ operator has more parameters
    # (separate alpha/beta orbital rotations plus a beta-beta block), so the actual per-walker
    # vector is longer. Re-size the (still-uninitialized, all-NaN) population array to match the
    # real trial-parameter width the first time we see it; on later iterations the array holds
    # real data and is left untouched.
    trial_width = int(np.asarray(trial_populations[0]).shape[0])
    if new_state.populations.shape[1] != trial_width and np.all(np.isnan(new_state.populations)):
        new_state.populations = np.full(
            (new_state.populations.shape[0], trial_width), np.nan, dtype=np.float64
        )

    best_index = int(np.nanargmin(trial_energies))
    if (
        current_state.best_energy() is None
        or trial_energies[best_index] < current_state.best_energy()
    ):
        # Update carryover when the best energy is updated
        logger.info(f"walker {best_index}: Update the best energy and carryover")
        new_state.best_index = best_index
        best_co_a, best_co_b = trial_carryovers[best_index]
        if best_co_a is not None:  # None when the walker produced no new pass (e.g. full resume)
            new_state.carryover, new_state.carryover_b = best_co_a, best_co_b
    for walker_idx in range(len(trial_energies)):
        if np.isnan(trial_energies[walker_idx]):
            continue
        delta_e = trial_energies[walker_idx] - current_state.energies[walker_idx]
        logger.info(
            f"walker {walker_idx}: Davidson final energy = "
            f"{trial_energies[walker_idx]} (ΔE = {delta_e})"
        )
        # Not in the source (behaviour change): a walker's first valid energy is accepted
        # whatever its sign. The source compared it with the initial 0.0, so a positive total
        # energy (a large constant term) was never accepted and the flow returned 0.0. A walker
        # with an all-NaN population has never been accepted.
        first_result = bool(np.all(np.isnan(new_state.populations[walker_idx])))
        if first_result or delta_e < 0:
            # Update reference energy and population when the trial gets lower energy
            new_state.energies[walker_idx] = trial_energies[walker_idx]
            new_state.populations[walker_idx] = trial_populations[walker_idx]
    return new_state


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    uhf_sqd_de.serve(
        name="uhf_sqd_de",
        description="UHF-SQD with LUCJ parameter optimization by differential evolution.",
    )
