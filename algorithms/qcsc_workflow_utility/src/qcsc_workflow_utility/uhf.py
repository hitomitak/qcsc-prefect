"""Unrestricted (UHF and broken-symmetry UHF) reference from an FCIDUMP.

The restricted counterpart is :mod:`qcsc_workflow_utility.chem`, which this module does not
change. Ported from the UHF-SQD work in kunalroy332/qcsc-prefect (branch
``bk/feature/bs-uhf-sqd-roquo`` at ``3968e585f7ca8137c127e5ee9f4fb383322d3d0d``,
``qcsc_workflow_utility/chem.py`` and ``active_space.py``). The computation is the source's
FCIDUMP path for ``unrestricted=True`` without an active-space manifest:

1. The FCIDUMP Hamiltonian is converged as a plain (non-symmetry-adapted) UHF.
2. Internal instabilities are followed, and an antiferromagnetic (AF) guess is also tried; the
   lower solution is kept (``_converge_broken_symmetry_uhf``).
3. The converged orbitals are cached by content (FCIDUMP sha256, guess, electron numbers).
4. The integrals are transformed to the alpha and beta MO bases, UCCSD gives the t2 blocks, and
   the diagonal of the UCCSD 1-RDM gives the initial occupancies.

The source read its options from environment variables (``FE4S4_AF_GROUPS``,
``FE4S4_AF_POL``, ``FE4S4_AF_FREE_S``, ``SBD_UHF_CACHE``, ``SBD_UHF_CACHE_DIR``,
``SEED_CCSD_MAX_CYCLE``, ``SEED_CCSD_RESTARTS``). Here they are explicit parameters of
:func:`compute_uhf_integrals_from_fcidump` with the same defaults, and no environment variable
is read. The ``fe4s4`` preset, the active-space manifest and orbital optimization are not
included.
"""

import contextlib
import hashlib
import io
import os
import re
import uuid
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from prefect import get_run_logger, task
from pydantic import BaseModel
from pyscf import ao2mo, cc, scf, tools

from .chem import NpStrict1DArrayF64, NpStrict2DArrayF64, NpStrict4DArrayF64


class UHFElectronicProperties(BaseModel):
    """Electronic properties of an unrestricted (UHF) reference.

    The alpha blocks are in the base fields (``one_body_tensor`` = h1 alpha,
    ``two_body_tensor`` = (aa|aa), ``t2`` = t2aa), as in
    :class:`~qcsc_workflow_utility.chem.ElectronicProperties`. The beta and mixed blocks are in
    the ``*_b``, ``*_ab`` and ``*_bb`` fields. All two-body tensors are in chemist notation;
    ``two_body_tensor_ab`` is (aa|bb).
    """

    one_body_tensor: NpStrict2DArrayF64
    two_body_tensor: NpStrict4DArrayF64
    t2: NpStrict4DArrayF64
    initial_occupancy: tuple[NpStrict1DArrayF64, NpStrict1DArrayF64]
    nuclear_repulsion_energy: float
    """The FCIDUMP constant term (ECORE)."""
    num_orbitals: int
    num_electrons: tuple[int, int]
    open_shell: bool
    spin_sq: float | None
    """Target value of the total spin squared passed to the solver, or None for no target."""

    unrestricted: bool = True

    one_body_tensor_b: NpStrict2DArrayF64
    two_body_tensor_ab: NpStrict4DArrayF64
    two_body_tensor_bb: NpStrict4DArrayF64
    t2_ab: NpStrict4DArrayF64
    t2_bb: NpStrict4DArrayF64

    # Provenance of a reference built from an active-space manifest in the source. Always None
    # here: the manifest path is not included. Kept so that records keep the source's keys.
    reference_tag: str | None = None
    reference_source: str | None = None
    fcidump_sha256: str | None = None

    # Diagnostics of the UHF reference itself. They do not enter any later computation.
    reference_energy: float
    """UHF total energy (``mf.e_tot``, including the constant term)."""
    reference_spin_square: float
    """<S^2> of the UHF determinant (``mf.spin_square()``); not the solver's target."""
    reference_cache_key: str
    """``v2|<FCIDUMP sha256>|<guess token>|nelec=(n_alpha, n_beta)``.

    Identifies the reference by the FCIDUMP contents, the broken-symmetry guess and the electron
    numbers. It does not include the code version. The cache file name is derived from it.
    """


def _get_logger():
    """Prefect run logger inside a flow/task, stdlib logger otherwise (direct calls, tests)."""
    try:
        return get_run_logger()
    except Exception:
        import logging

        return logging.getLogger(__name__)


def file_sha256(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """sha256 hex digest of a file, streamed in 1 MiB chunks (FCIDUMPs can be ~100 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_fcidump_header(path: str | os.PathLike) -> dict[str, str]:
    """Header keys and values of an FCIDUMP, tokenized as ``pyscf.tools.fcidump.read`` does.

    Only the header is read (at most 10 lines, up to ``&END`` or ``/``); keys and values are
    upper-case strings without spaces or trailing commas.
    """
    lines = []
    with open(path) as f:
        for _ in range(10):
            line = f.readline().upper()
            lines.append(line)
            if "&END" in line or "/" in line:
                break
        else:
            raise ValueError(f"{path}: no FCIDUMP header end (&END or /) in the first 10 lines")
    tokens = ",".join(lines).replace("&FCI", "").replace("&END", "").replace("/", "")
    tokens = tokens.replace(" ", "").replace("\n", "").replace(",,", ",")
    header = {}
    for token in re.split(",(?=[a-zA-Z])", tokens):
        if "=" in token:
            key, value = token.split("=", 1)
            header[key] = value.strip(",")
    return header


def _check_restricted_fcidump(path: str | os.PathLike) -> None:
    """Refuse an FCIDUMP that is not one restricted (spin-shared) spatial-orbital Hamiltonian.

    The UHF reference is built from a restricted FCIDUMP. Two other layouts would be read by
    ``pyscf.tools.fcidump`` into a different Hamiltonian without an error:

    - a MOLPRO-style UHF FCIDUMP (``IUHF`` other than 0), whose alpha, beta and mixed blocks
      overwrite each other in one restricted array;
    - the interleaved spin-orbital FCIDUMP for the SBD ``-D_UHF`` driver
      (``QCSC_INTEGRALS=SBD_UHF_INTERLEAVED``, written by ``qcsc_prefect_sbd``).

    ``IUHF=0`` (MOLPRO's restricted flag) and ``QCSC_INTEGRALS=RESTRICTED`` are accepted.
    """
    header = _read_fcidump_header(path)
    if "IUHF" in header:
        value = header["IUHF"]
        try:
            restricted = int(value) == 0
        except ValueError:
            restricted = False
        if not restricted:
            raise ValueError(
                f"{path}: IUHF={value} is a MOLPRO-style UHF FCIDUMP. The UHF reference needs a "
                "restricted spatial-orbital FCIDUMP (no IUHF, or IUHF=0)."
            )
    if "QCSC_INTEGRALS" in header and header["QCSC_INTEGRALS"] != "RESTRICTED":
        raise ValueError(
            f"{path}: QCSC_INTEGRALS={header['QCSC_INTEGRALS']} is not a restricted FCIDUMP. "
            "The UHF reference needs a restricted spatial-orbital FCIDUMP (no QCSC_INTEGRALS, "
            "or QCSC_INTEGRALS=RESTRICTED)."
        )


AF_OPTION_KEYS = ("up", "down", "free", "pol")
"""Keys of an ``af_groups`` mapping that are options, not fragments."""


def _validate_orbital_indices(
    af_groups: Mapping[str, Any], norb: int, *, source: str = "af_groups"
) -> None:
    """Raise ``ValueError`` if any fragment orbital index in an af_groups dict is outside
    ``[0, norb)`` or not an integer. Option keys (``up``/``down``/``free``/``pol``) are skipped."""
    for name, orbs in af_groups.items():
        if name in AF_OPTION_KEYS:
            continue
        if not isinstance(orbs, (list, tuple, np.ndarray)) or any(
            isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in orbs
        ):
            raise ValueError(f"{source}: fragment {name!r} has non-integer orbitals {orbs!r}")
        bad = [int(x) for x in orbs if not 0 <= x < norb]
        if bad:
            raise ValueError(
                f"{source}: fragment {name!r} orbital indices {bad} out of range for NORB={norb} "
                f"(valid 0..{norb - 1})"
            )


def _density_from_af_groups(af_groups: Mapping[str, Any], norb: int) -> np.ndarray:
    """Diagonal ``(2, norb, norb)`` guess density from an af_groups dict.

    Same fill as the atom-localized guess in ``_converge_broken_symmetry_uhf``: up/down
    fragments split ``0.5 +/- 0.5*pol``, ``free`` fragments 0.5/0.5, all others doubly occupied.
    Orbitals in no fragment stay empty. Indices are bounds-checked first.
    """
    _validate_orbital_indices(af_groups, norb)
    up = set(af_groups.get("up", []))
    down = set(af_groups.get("down", []))
    free = set(af_groups.get("free", []))
    pol = float(af_groups.get("pol", 1.0))
    dm0 = np.zeros((2, norb, norb))
    for name, orbs in af_groups.items():
        if name in AF_OPTION_KEYS:
            continue
        for x in orbs:
            if name in up:
                dm0[0, x, x] = 0.5 + 0.5 * pol
                dm0[1, x, x] = 0.5 - 0.5 * pol
            elif name in down:
                dm0[0, x, x] = 0.5 - 0.5 * pol
                dm0[1, x, x] = 0.5 + 0.5 * pol
            elif name in free:
                dm0[:, x, x] = 0.5
            else:
                dm0[:, x, x] = 1.0
    return dm0


def _af_guess_spec(af_groups: Mapping[str, Any] | None) -> dict | None:
    """The source's AF-guess spec from an explicit ``af_groups`` mapping, or None.

    ``af_groups`` maps fragment names to lists of orbital indices (0-based). ``up`` and
    ``down`` list the fragments seeded spin-up and spin-down; ``free`` lists fragments seeded
    half-filled and unpolarized; all other fragments are doubly occupied. ``pol`` (default 1.0)
    is the polarization amplitude: up/down fragments get ``0.5 +/- 0.5 * pol``.

    As in the source (``FE4S4_AF_GROUPS`` plus ``FE4S4_AF_POL``), the returned spec always has
    ``pol``, so the guess token of the cache key matches the source's for the same settings.
    """
    if af_groups is None:
        return None
    if not isinstance(af_groups, Mapping):
        raise ValueError(f"af_groups must be a mapping, got {type(af_groups).__name__}")
    spec = dict(af_groups)
    spec["pol"] = float(spec.get("pol", 1.0))
    return spec


def _check_af_groups(af_groups: dict | None, norb: int) -> None:
    """Bounds-check an AF-guess spec against the FCIDUMP's NORB before any SCF runs."""
    if not af_groups:
        return
    _validate_orbital_indices(af_groups, norb)


def _converge_broken_symmetry_uhf(mf, max_follow: int = 5, af_groups: dict | None = None):
    """Drive a UHF calculation to the genuine (possibly spin-broken) solution.

    For a closed-shell singlet, plain UHF started from the RHF density stays at the RHF solution
    (the RHF determinant is a stationary point of the UHF equations) -- so ``mf.to_uhf().kernel()``
    returns an *effectively restricted* reference (<S^2> = 0, identical energy). For strongly
    correlated systems (e.g. Fe-S clusters) the true UHF is lower in energy and spin-broken; seeding
    the LUCJ ansatz / UCCSD amplitudes from the unbroken solution is a restricted reference in
    disguise and defeats the point of running UHF.

    This runs UHF, then iteratively performs an internal STABILITY analysis: if the current solution
    is unstable, it rebuilds the density from the lowest instability mode and re-converges (via
    Newton), following the instability down to a stable (broken-symmetry when appropriate) minimum.
    This is the standard PySCF recipe for reaching the true UHF solution.

    ``af_groups`` is the spec from :func:`_af_guess_spec` (None: energy-ordered AF guess only).

    IMPORTANT (Fe-S cubanes): for [4Fe-4S] the internal stability analysis from a SYMMETRIC guess
    stays in the spin-pure (RHF) basin (<S^2>=0, -326.547) -- it does NOT find the antiferromagnetic
    broken-symmetry minimum, because breaking the spatial symmetry to localize the Fe spins
    requires a spin-IMBALANCED initial density, not just following an internal instability. So we
    FIRST try an antiferromagnetic guess (promote a few beta electrons across the HOMO-LUMO gap, per
    the Noodleman BS recipe), stability-follow that, and keep whichever solution (default vs
    AF-guess) is LOWER. For Fe4S4 the AF guess reaches -326.767 (<S^2>~7.6), 219 mHa below the
    spin-pure solution.
    """

    def _follow_stability(m, n=max_follow):
        stab = False
        for _ in range(n):
            new_mo = m.stability()[0]
            cur = m.mo_coeff
            if np.allclose(new_mo[0], cur[0]) and np.allclose(new_mo[1], cur[1]):
                stab = True
                break
            m = m.newton().run(m.make_rdm1(new_mo, m.mo_occ))
        return m, stab

    def _follow_stability_external(m, n=10):
        """Follow BOTH internal and EXTERNAL instabilities and keep the LOWEST iterate seen.

        The Noodleman AF sublattice minima (Singlet-I/II/III) are near-degenerate and separated by
        external (spin-symmetry-breaking) instabilities that plain internal following (_follow_
        stability) does not cross -- it declares "stable" at the first internal minimum (~-326.769
        for Fe4S4) and never reaches the lower external basin (~-326.801). Mirrors the
        collaborator recipe: m.stability(return_status=True, external=True), rebuild, repeat,
        track the minimum."""
        best = m
        for _ in range(n):
            res = m.stability(return_status=True, external=True)
            mo = res[0]
            stable_i, stable_e = res[-2], res[-1]
            try:
                # Pass the (external-)instability MO coefficients DIRECTLY to the Newton solver as
                # the initial guess (not a density rebuilt from them): newton().run(mo_coeff)
                # follows the external spin-symmetry-breaking mode into the lower AF basin, whereas
                # run(make_rdm1(mo)) stalls at the higher internal minimum. This is the collaborator
                # recipe and is what reaches -326.801 (vs -326.771 with the density rebuild).
                m = m.newton().run(mo if isinstance(mo, tuple) else m.mo_coeff)
            except Exception:
                break
            if m.e_tot < best.e_tot:
                best = m
            if stable_i and stable_e:
                break
        return best, True

    def _af_guess_uhf(base, base_mol, base_hcore, base_ovlp, base_eri, base_nuc):
        """A fresh plain scf.UHF (same integrals/mol) started from an AF spin-imbalanced density.
        Returns the stability-followed result, or None if it fails. Tries a couple of promotion
        widths. Built from a PLAIN scf.UHF on the mol (not base.__class__, which may be a
        newton-wrapped SecondOrderUHF lacking a simple mol constructor)."""
        try:
            na, nb = base_mol.nelec
        except Exception:
            return None
        n = base_mol.nao
        best = None
        for shift in (2, 1, 3):
            if nb - shift < 0 or nb + shift > n:
                continue
            m = scf.UHF(base_mol)
            # carry the FCIDUMP integral overrides if present
            if base_hcore is not None:
                m.get_hcore = lambda *a, **k: base_hcore
            if base_ovlp is not None:
                m.get_ovlp = lambda *a, **k: base_ovlp
            if base_eri is not None:
                m._eri = base_eri
            if base_nuc is not None:
                m.mol.energy_nuc = lambda *a: base_nuc
            m.max_cycle = 400
            m.conv_tol = 1e-9
            mo = np.eye(n)
            occ_a = list(range(na))
            occ_b = list(range(nb - shift)) + list(range(nb, nb + shift))
            dma = mo[:, occ_a] @ mo[:, occ_a].T
            dmb = mo[:, occ_b] @ mo[:, occ_b].T
            try:
                m.kernel(dm0=(dma, dmb))
                m, _ = _follow_stability(m)
                if best is None or m.e_tot < best.e_tot:
                    best = m
            except Exception:
                continue

        # Atom-localized (Noodleman) AF guess: when an orbital->fragment map is supplied, seed the
        # density so specific magnetic centers get spin-UP and the counter-set spin-DOWN (the true
        # antiferromagnetic sublattices), instead of a chemistry-blind energy-ordered promotion. For
        # Fe-S cubanes this lands in a ~33 mHa LOWER broken-symmetry basin than the energy-ordered
        # guess (Fe4S4: -326.801 vs -326.768) -- the reference orbitals span the Fe-3d correlation
        # far better, which is what SQD actually recovers on. `af_groups` maps fragment name ->
        # list[orbital idx]; keys "up"/"down" list which fragments are alpha/beta-polarized, closed
        # fragments (doubly occupied) are everything else. We stability-follow (external) and keep
        # the lowest across all guesses.
        if af_groups:
            # Fractional polarization amplitude p in [0,1]: p=1 -> full 1.0/0.0 spin split on the
            # magnetic centers (maximally localized, large <S^2>); p<1 -> 0.5(1+/-p) split, letting
            # the sublattices couple more (SMALLER <S^2>, closer to the physical super-exchange
            # regime). The paper notes the singlet-triplet exchange runs through the bridging S 3p
            # orbitals -- so "free" fragments (the bridging S) are seeded half-filled to MEDIATE
            # that exchange rather than being forced doubly-occupied. Local sweep: p=0.6 + free
            # bridging S gives the deepest BS-UCCSD (-327.143 vs -327.102 at p=1.0) and lowest
            # <S^2>.
            # The fill (up/down 0.5 +/- 0.5*pol, free 0.5/0.5, other fragments doubly occupied)
            # lives in _density_from_af_groups, which bounds-checks the indices first.
            dm0 = _density_from_af_groups(af_groups, n)
            m = scf.UHF(base_mol)
            if base_hcore is not None:
                m.get_hcore = lambda *a, **k: base_hcore
            if base_ovlp is not None:
                m.get_ovlp = lambda *a, **k: base_ovlp
            if base_eri is not None:
                m._eri = base_eri
            if base_nuc is not None:
                m.mol.energy_nuc = lambda *a: base_nuc
            m.max_cycle = 400
            m.conv_tol = 1e-9
            try:
                m.kernel(dm0=(dm0[0], dm0[1]))
                # Follow external instabilities (10 rounds): the AF sublattice minima are
                # near-degenerate (Singlet-I/II/III) and only EXTERNAL following reaches the lowest
                # basin (~-326.801 for Fe4S4); internal-only following stalls at ~-326.769.
                m, _ = _follow_stability_external(m, n=10)
                if best is None or m.e_tot < best.e_tot:
                    best = m
            except Exception:
                pass
        return best

    # Fail fast on an af_groups spec that does not fit this Hamiltonian (e.g. the 36-orbital
    # 'fe4s4' preset on a 16-orbital AVAS FCIDUMP), before any SCF work.
    _check_af_groups(af_groups, mf.mol.nao)

    # Capture the mol + any FCIDUMP integral overrides from the ORIGINAL mf before newton-wrapping,
    # so the AF-guess UHF can be rebuilt as a plain scf.UHF with the same Hamiltonian.
    _base_mol = mf.mol
    _base_hcore = mf.get_hcore() if callable(getattr(mf, "get_hcore", None)) else None
    _base_ovlp = mf.get_ovlp() if callable(getattr(mf, "get_ovlp", None)) else None
    _base_eri = getattr(mf, "_eri", None)
    try:
        _base_nuc = float(mf.mol.energy_nuc())
    except Exception:
        _base_nuc = None

    mf.kernel()
    stable = False
    for _ in range(max_follow):
        # stability() returns the (internal) stable MOs; if they differ from the current MOs the
        # solution was unstable and we follow them down.
        new_mo = mf.stability()[0]
        # new_mo is (mo_a, mo_b); detect whether it changed (instability found).
        cur = mf.mo_coeff
        changed = not (np.allclose(new_mo[0], cur[0]) and np.allclose(new_mo[1], cur[1]))
        if not changed:
            stable = True
            break  # stable -> genuine UHF minimum reached
        dm = mf.make_rdm1(new_mo, mf.mo_occ)
        mf = mf.newton().run(dm)

    # Also try the antiferromagnetic broken-symmetry guess and keep whichever solution is LOWER.
    # For closed-shell/RHF-basin cases the AF guess just returns to the same energy (no harm); for
    # Fe-S cubanes it finds the genuine BS minimum the stability-only path misses.
    af = _af_guess_uhf(mf, _base_mol, _base_hcore, _base_ovlp, _base_eri, _base_nuc)
    if af is not None and af.converged and af.e_tot < mf.e_tot - 1e-6:
        try:
            ss_af, _ = af.spin_square()
        except Exception:
            ss_af = float("nan")
        warnings.warn(
            f"Broken-symmetry UHF found via AF guess: E={af.e_tot:.6f} <S^2>={ss_af:.3f} is "
            f"{(mf.e_tot - af.e_tot) * 1000:.1f} mHa below the stability-only solution "
            f"({mf.e_tot:.6f}). Using the broken-symmetry reference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return af

    # Robustness (audit P4): if the follow loop exhausted max_follow while still unstable, the
    # returned reference is NOT a genuine minimum. Warn loudly and report the final <S^2> so a
    # downstream seed built from a non-stable / spin-contaminated reference is visible, not silent.
    if not stable:
        try:
            ss, _mult = mf.spin_square()
        except Exception:
            ss = float("nan")
        warnings.warn(
            f"UHF stability following did not converge to a stable solution after "
            f"{max_follow} follows; returning the last iterate (<S^2>={ss:.4f}). The UHF "
            f"reference may be non-stationary or spin-contaminated -- t2/occupancy seeds built "
            f"from it should be treated with caution.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mf


def _fcidump_uhf(mf, nuc: float):
    """Plain (non-symmetry-adapted) ``scf.UHF`` on the FCIDUMP Hamiltonian carried by the RHF
    shell ``mf`` from ``tools.fcidump.to_scf`` (see the Audit P3 note at the call site)."""
    base_mol = mf.mol
    if getattr(base_mol, "symmetry", False):
        base_mol.symmetry = False
        base_mol.build(dump_input=False, parse_arg=False)
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = mf._eri
    out = scf.UHF(base_mol)
    out.get_hcore = lambda *args, **kwargs: hcore
    out.get_ovlp = lambda *args, **kwargs: ovlp
    out._eri = eri
    out.mol.energy_nuc = lambda *args: nuc  # preserved across the conversion
    out.symmetry = False
    return out


# Bump when the cached content or key recipe changes.
_UHF_CACHE_VERSION = "v2"

DEFAULT_UHF_CACHE_DIR = "~/.cache/qcsc_uhf"
"""The source's default cache directory (``SBD_UHF_CACHE_DIR`` unset)."""


def _uhf_cache_key(fcidump_sha256: str, guess_token: str, nelec) -> str:
    """The raw cache key ``v2|<FCIDUMP sha256>|<guess token>|nelec=(n_alpha, n_beta)``."""
    nelec_txt = "nelec=" + str(tuple(int(n) for n in nelec))
    return "|".join([_UHF_CACHE_VERSION, fcidump_sha256, guess_token, nelec_txt])


def _uhf_cache_path(cache_dir: str, fcidump_sha256: str, guess_token: str, nelec) -> str:
    """Content-based cache file: FCIDUMP sha256 + guess token + (n_alpha, n_beta).

    The FCIDUMP path is deliberately not part of the key: the same file copied elsewhere hits,
    a regenerated file at the same path misses.
    """
    raw = _uhf_cache_key(fcidump_sha256, guess_token, nelec)
    key = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]
    return os.path.join(cache_dir, f"uhf_bs_{key}.npz")


def _uhf_cache_load(cpath: str, mf, norb: int) -> bool:
    """Load a cached reference into ``mf`` (mo_coeff / mo_occ / mo_energy / e_tot). Returns
    False (and leaves ``mf`` untouched) if absent, unreadable, or inconsistent with ``mf``."""
    if not os.path.exists(cpath):
        return False
    try:
        with np.load(cpath, allow_pickle=False) as c:
            data = {k: c[k] for k in c.files}
        nelec = tuple(int(n) for n in data["nelec"])
        shapes_ok = all(data[k].shape == (norb, norb) for k in ("mo_a", "mo_b"))
        occ = (int(round(data["occ_a"].sum())), int(round(data["occ_b"].sum())))
    except Exception as exc:
        _get_logger().info(f"[uhf-cache] ignoring unreadable cache {cpath}: {exc}")
        return False
    if nelec != tuple(mf.mol.nelec) or occ != nelec or not shapes_ok:
        _get_logger().info(
            f"[uhf-cache] ignoring {cpath}: cached nelec={nelec} occ={occ} norb ok={shapes_ok} "
            f"do not match nelec={tuple(mf.mol.nelec)} norb={norb}"
        )
        return False
    mf.mo_coeff = (data["mo_a"], data["mo_b"])
    mf.mo_occ = (data["occ_a"], data["occ_b"])
    mf.mo_energy = (data["ene_a"], data["ene_b"])
    mf.e_tot = float(data["e_tot"])
    mf.converged = True
    tag = f" tag={data['tag']}" if "tag" in data else ""
    _get_logger().info(f"[uhf-cache] reuse BS-UHF reference {cpath} (E={mf.e_tot:.6f}{tag})")
    return True


def _uhf_cache_save(cpath: str, mf, *, tag: str | None = None, source: str | None = None) -> None:
    """Write the converged reference atomically (tmp file + rename); failures only warn.

    Each write has its own temporary file in the cache directory, so concurrent writers of the
    same reference (threads of one process, or processes) never share one; the last rename wins.
    """
    tmp = None
    try:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        mo_a, mo_b = mf.mo_coeff
        occ_a, occ_b = mf.mo_occ
        ene_a, ene_b = mf.mo_energy
        extra = {k: np.array(v) for k, v in (("tag", tag), ("source", source)) if v is not None}
        tmp = f"{cpath}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "xb") as f:
            np.savez(
                f,
                mo_a=mo_a,
                mo_b=mo_b,
                occ_a=occ_a,
                occ_b=occ_b,
                ene_a=ene_a,
                ene_b=ene_b,
                e_tot=float(mf.e_tot),
                nelec=np.array(mf.mol.nelec),
                **extra,
            )
        os.replace(tmp, cpath)
        _get_logger().info(f"[uhf-cache] saved BS-UHF reference {cpath} (E={mf.e_tot:.6f})")
    except Exception as exc:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.remove(tmp)
        _get_logger().warning(f"[uhf-cache] save failed (continuing): {exc}")


def _build_property_uhf(
    mf: scf.UHF,
    norb: int,
    spin_sq: float | None,
    buf: io.StringIO,
    reference_cache_key: str,
    ccsd_max_cycle: int = 50,
    ccsd_restarts: int = 10,
) -> UHFElectronicProperties:
    """Build ElectronicProperties from an unrestricted (UHF) calculation.

    Mirrors ``chem._build_property`` but keeps separate alpha/beta MO coefficients, three
    two-body integral blocks (aa, ab, bb), the UCCSD t2 tuple (t2aa, t2ab, t2bb), and per-spin
    occupancies. The alpha blocks go into the base ElectronicProperties fields; beta / mixed
    blocks go into the optional *_b / *_ab / *_bb fields.

    Args:
        mf: PySCF UHF object after calculation.
        norb: Number of (spatial) orbitals.
        spin_sq: Target value for the total spin squared, or None for no target.
        buf: StringIO buffer containing PySCF logs.
        reference_cache_key: The reference's cache key (see ``UHFElectronicProperties``).
        ccsd_max_cycle: UCCSD cycles per run (the source's ``SEED_CCSD_MAX_CYCLE``).
        ccsd_restarts: Number of UCCSD runs; each restart continues from the last amplitudes
            with a fresh DIIS (the source's ``SEED_CCSD_RESTARTS``).

    Returns:
        UHFElectronicProperties object.
    """
    mo_a, mo_b = mf.mo_coeff
    hcore = mf.get_hcore()

    # One-body integrals in each spin's MO basis.
    h1_a = mo_a.T @ hcore @ mo_a
    h1_b = mo_b.T @ hcore @ mo_b

    # Two-body integrals: alpha-alpha, beta-beta, and the mixed alpha-beta block.
    eri = mf._eri
    h2_aa = ao2mo.full(eri, mo_a, compact=False).reshape(norb, norb, norb, norb)
    h2_bb = ao2mo.full(eri, mo_b, compact=False).reshape(norb, norb, norb, norb)
    h2_ab = ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape(
        norb, norb, norb, norb
    )

    nuclear_repulsion_energy = mf.mol.energy_nuc()
    num_elec_a, num_elec_b = mf.mol.nelec

    # Run UCCSD; t2 is the tuple (t2aa, t2ab, t2bb). Strongly-correlated Fe-S active spaces often
    # do not converge in one 50-cycle run. Restart from the current amplitudes (fresh DIIS) up to
    # ccsd_restarts times instead of one long run (cycles per run: ccsd_max_cycle).
    mycc = cc.UCCSD(mf)
    mycc.max_cycle = int(ccsd_max_cycle)
    mycc.diis_space = 12
    mycc.kernel()
    n_runs = max(1, int(ccsd_restarts))
    for _ in range(n_runs - 1):
        if mycc.converged:
            break
        mycc.kernel(mycc.t1, mycc.t2)
    if not mycc.converged:
        _get_logger().warning(
            "UCCSD seed not converged after %d run(s) x %d cycles; using the last amplitudes.",
            n_runs,
            mycc.max_cycle,
        )
    t2_aa, t2_ab, t2_bb = mycc.t2

    # Per-spin natural-orbital occupancies from the *correlated* UCCSD 1-RDM (in MO basis).
    # These must be fractional: configuration recovery downstream uses them to bias sampled
    # bitstrings toward the physically important (near-HF) configurations. The UHF *SCF* RDM is
    # (often) idempotent -> integer 0/1 occupancies, which give recovery no signal and collapse
    # the SQD subspace to bare Hartree-Fock. UCCSD make_rdm1() returns (dm_a, dm_b) in MO basis.
    # Use the DIAGONAL (per-canonical-MO occupancy) of the correlated UCCSD 1-RDM, not the sorted
    # eigenvalues. The bitstrings are measured in the canonical MO basis (PrepareHartreeFockJW +
    # integrals from mf.mo_coeff), so recover_configurations needs occ of MO p in THAT basis; the
    # eigenvalues are natural-orbital occupation numbers in a rotated basis and, once sorted, would
    # misassign the bias to the wrong orbitals (the error lands on the active, partially-occupied
    # orbitals and grows with system size).
    dm_cc_a, dm_cc_b = mycc.make_rdm1()
    occ_a = np.diag(dm_cc_a)
    occ_b = np.diag(dm_cc_b)

    # Get PySCF logs dumped into in-memory buffer
    _get_logger().info(buf.getvalue())

    return UHFElectronicProperties(
        one_body_tensor=h1_a,
        two_body_tensor=h2_aa,
        t2=t2_aa,
        one_body_tensor_b=h1_b,
        two_body_tensor_ab=h2_ab,
        two_body_tensor_bb=h2_bb,
        t2_ab=t2_ab,
        t2_bb=t2_bb,
        initial_occupancy=(occ_a, occ_b),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        unrestricted=True,
        spin_sq=spin_sq,
        reference_energy=float(mf.e_tot),
        reference_spin_square=float(mf.spin_square()[0]),
        reference_cache_key=reference_cache_key,
    )


@task(
    persist_result=True,
    result_serializer="compressed/json",
    name="compute_uhf_molecular_integrals",
    # One stored result per task run. With the default cache policy, Prefect would return the
    # result of an earlier call with the same FCIDUMP path in the same flow run, even after the
    # file changed, before this function (and its content hash) is reached.
    result_storage_key="uhf-reference/{task_run.id}",
)
def compute_uhf_integrals_from_fcidump(
    fcidump_file: str,
    spin_sq: float | None = 0.0,
    *,
    af_groups: Mapping[str, Any] | None = None,
    cache_dir: str | None = DEFAULT_UHF_CACHE_DIR,
    cache_read: bool = True,
    ccsd_max_cycle: int = 50,
    ccsd_restarts: int = 10,
) -> UHFElectronicProperties:
    """UHF (broken-symmetry when it is lower) reference and integrals from an FCIDUMP.

    The FCIDUMP is restricted and spatial, with all orbitals active; its ``MS2`` header sets
    ``n_alpha - n_beta``, and its constant term becomes ``nuclear_repulsion_energy``. A UHF
    FCIDUMP (``IUHF`` other than 0, or ``QCSC_INTEGRALS`` other than ``RESTRICTED``) raises
    ``ValueError`` before any integral is read.

    Each call computes a fresh result: Prefect results are stored per task run
    (``result_storage_key``), and are never reused by input, because the inputs name the
    FCIDUMP by path and not by content. Reuse of a converged reference is the content-keyed
    ``cache_dir`` cache.

    Args:
        fcidump_file: Location of the FCIDUMP file.
        spin_sq: Target value for the total spin squared passed on to the solver, or None for
            no target (required by the SBD solver). Not used here.
        af_groups: Atom-localized (Noodleman) AF guess, as a mapping of fragment names to
            0-based orbital indices with the option keys ``up``, ``down``, ``free`` and ``pol``;
            see :func:`_af_guess_spec`. None (default) tries only the energy-ordered AF guess.
            The source read it from ``FE4S4_AF_GROUPS`` (without its ``fe4s4`` preset),
            ``FE4S4_AF_POL`` and ``FE4S4_AF_FREE_S``.
        cache_dir: Directory of the content-keyed reference cache (the source's
            ``SBD_UHF_CACHE_DIR``); ``~`` is expanded. None disables reading and writing the
            cache.
        cache_read: Whether a cached reference is reused (the source's ``SBD_UHF_CACHE``). A
            fresh result is written to the cache either way.
        ccsd_max_cycle: UCCSD cycles per run (the source's ``SEED_CCSD_MAX_CYCLE``).
        ccsd_restarts: Number of UCCSD runs (the source's ``SEED_CCSD_RESTARTS``).

    Returns:
        UHFElectronicProperties object.
    """
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    # Before reading the integrals: a UHF-layout FCIDUMP would be read without an error.
    _check_restricted_fcidump(fcidump_file)
    data = tools.fcidump.read(fcidump_file)
    norb = data["NORB"]

    mf = tools.fcidump.to_scf(fcidump_file)
    mf.mol.verbose = 4
    mf.mol.stdout = buf

    # pyscf 2.13 has no fcidump.to_uhf_scf; convert the RHF shell (which already carries the
    # FCIDUMP integral overrides and mol.spin = MS2) into a UHF object and run it. UHF init
    # guess builds two spin densities from nelec=(na, nb); no manual dm0 needed.
    nuc = mf.mol.energy_nuc()
    # Audit P3: tools.fcidump.to_scf sets mol.symmetry=True with a *guessed* point group when
    # the FCIDUMP carries ORBSYM. Disable symmetry on the mol, rebuild it, and construct a PLAIN
    # scf.UHF on it (not mf.to_uhf(), which keeps the symmetry-adapted class); carry the
    # FCIDUMP integral overrides (get_hcore/_eri) + nuclear energy across by hand.
    mf = _fcidump_uhf(mf, nuc)
    # Deterministic broken-symmetry reference: cache converged UHF MOs so every run starts from
    # the IDENTICAL reference. Hard cases (Fe4S4's 400-cycle SCF) are thread/BLAS-order
    # sensitive -> different BS local minima run-to-run otherwise. The key is content-based
    # (FCIDUMP sha256 + guess token + nelec), so a regenerated FCIDUMP at the same path is never
    # served a stale reference. cache_read=False skips the lookup (the fresh result still
    # refreshes the cache file).
    _af_groups = _af_guess_spec(af_groups)
    _check_af_groups(_af_groups, norb)  # before the cache lookup: clear error, not IndexError
    guess_token = "af=" + str(_af_groups)
    fcidump_sha256 = file_sha256(fcidump_file)
    reference_cache_key = _uhf_cache_key(fcidump_sha256, guess_token, mf.mol.nelec)
    if cache_dir is None:
        mf = _converge_broken_symmetry_uhf(mf, af_groups=_af_groups)
    else:
        cpath = _uhf_cache_path(
            os.path.expanduser(cache_dir), fcidump_sha256, guess_token, mf.mol.nelec
        )
        if not (cache_read and _uhf_cache_load(cpath, mf, norb)):
            mf = _converge_broken_symmetry_uhf(mf, af_groups=_af_groups)
            _uhf_cache_save(cpath, mf)
    return _build_property_uhf(
        mf,
        norb,
        spin_sq,
        buf,
        reference_cache_key,
        ccsd_max_cycle=ccsd_max_cycle,
        ccsd_restarts=ccsd_restarts,
    )
