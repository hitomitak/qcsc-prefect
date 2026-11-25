"""Run the -D_UHF SBD driver on small UHF problems and check it against PySCF FCI.

Set SBD_NATIVE_UHF_BINARY to a driver built with ``UHF=1`` (``native/diag_uhf``) and
SBD_NATIVE_BINARY to the restricted driver (``native/diag``); SBD_NATIVE_LAUNCHER and
SBD_NATIVE_OMP_THREADS as in test_sbd_native.py. For the GPU drivers, use ``native/diag-gpu_uhf``
and ``native/diag-gpu`` with several threads. Normal Python-only installations skip these tests.

The reference energy is PySCF FCI on the restricted (RO)HF integrals of the same molecule.
With every orbital active, full-space FCI does not depend on the orbital basis, so the UHF
transform, the interleaved FCIDUMP, the determinant files and the constant term are all
checked against a calculation that uses none of them.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import cistring, direct_uhf
from qcsc_prefect_sbd import solver_job

UHF_BINARY = os.environ.get("SBD_NATIVE_UHF_BINARY")
RHF_BINARY = os.environ.get("SBD_NATIVE_BINARY")
OMP_THREADS = os.environ.get("SBD_NATIVE_OMP_THREADS") or "1"
pytestmark = pytest.mark.skipif(
    not (UHF_BINARY and RHF_BINARY),
    reason="Set SBD_NATIVE_UHF_BINARY and SBD_NATIVE_BINARY to test the UHF SBD driver",
)

# Tight Davidson settings: the comparison is with exact FCI, so the solve must converge.
TIGHT = dict(block=30, iteration=60, tolerance=1e-10)

MOLECULES = {
    # n_alpha != n_beta: NH2 doublet, 7 orbitals, (5, 4) electrons, 21 x 35 determinants.
    "nh2_doublet": dict(atom="N 0 0 0; H 0 0.80 0.59; H 0 -0.80 0.59", spin=1, broken=False),
    # n_alpha == n_beta broken-symmetry UHF: stretched H6 chain, (3, 3), <S^2> about 2.74.
    "h6_broken_symmetry": dict(
        atom="; ".join(f"H 0 0 {2.0 * i}" for i in range(6)), spin=0, broken=True
    ),
}


@pytest.fixture(autouse=True)
def quiet_logger(monkeypatch):
    logger = SimpleNamespace(
        debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: logger)


def converged_uhf(mol, broken):
    """UHF following internal instabilities; a perturbed start breaks the spin symmetry."""
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    if broken:
        dma, dmb = (dm.copy() for dm in mf.get_init_guess())
        half = mol.nao // 2
        dma[:half, :half] *= 1.3
        dmb[half:, half:] *= 1.3
        mf.kernel(dm0=(dma, dmb))
    else:
        mf.kernel()
    for _ in range(10):
        mo, stable, _, _ = mf.stability(return_status=True)
        if stable:
            break
        mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
    assert mf.converged
    return mf


def uhf_problem(name):
    spec = MOLECULES[name]
    mol = gto.M(atom=spec["atom"], basis="sto-3g", spin=spec["spin"], verbose=0)
    mf = converged_uhf(mol, spec["broken"])
    ca, cb = mf.mo_coeff
    hcore = mf.get_hcore()
    n = mol.nao
    problem = SimpleNamespace(
        mol=mol,
        mf=mf,
        norb=n,
        nelec=mol.nelec,
        h1_a=ca.T @ hcore @ ca,
        h1_b=cb.T @ hcore @ cb,
        h2_aa=ao2mo.restore(1, ao2mo.full(mol, ca), n),
        h2_ab=ao2mo.general(mol, (ca, ca, cb, cb), compact=False).reshape((n,) * 4),
        h2_bb=ao2mo.restore(1, ao2mo.full(mol, cb), n),
    )
    # Reference: FCI with restricted (RO)HF orbitals, independent of the UHF integrals.
    rmf = (scf.ROHF(mol) if spec["spin"] else scf.RHF(mol)).run(conv_tol=1e-12)
    solver = fci.FCI(mol, rmf.mo_coeff)
    solver.conv_tol = 1e-12
    problem.e_fci = solver.kernel()[0]
    return problem


def all_strings(norb, n_elec):
    hartree_fock = (1 << n_elec) - 1
    strings = sorted(int(s) for s in cistring.make_strings(range(norb), n_elec))
    return np.array([hartree_fock] + [s for s in strings if s != hartree_fock], dtype=np.int64)


def run_driver(binary, work_dir, args, ranks=1):
    launcher = shlex.split(os.environ.get("SBD_NATIVE_LAUNCHER", "mpiexec"))
    command = launcher + ["-n", str(ranks), str(Path(binary).resolve())] + args
    environment = dict(os.environ, OMP_NUM_THREADS=OMP_THREADS, OPENBLAS_NUM_THREADS="1")
    return subprocess.run(
        command, cwd=work_dir, env=environment, capture_output=True, text=True, timeout=300
    )


def prep_unrestricted(work_dir, p, alpha=None, beta=None):
    solver_job._prep_files(
        work_dir=work_dir,
        ci_strings=(
            all_strings(p.norb, p.nelec[0]) if alpha is None else alpha,
            all_strings(p.norb, p.nelec[1]) if beta is None else beta,
        ),
        one_body_tensor=p.h1_a,
        two_body_tensor=p.h2_aa,
        norb=p.norb,
        nelec=p.nelec,
        beta_integrals=(p.h1_b, p.h2_ab, p.h2_bb),
    )


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("name", list(MOLECULES))
def test_uhf_driver_full_space_energy_matches_fci(tmp_path, name, ranks):
    p = uhf_problem(name)
    prep_unrestricted(tmp_path, p)
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), integrals="unrestricted", **TIGHT)
    run = run_driver(
        UHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True), ranks
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = solver_job._read_files(
        work_dir=tmp_path, norb=p.norb, separate_beta=True, integrals="unrestricted"
    )
    assert result.integrals == "SBD_UHF_INTERLEAVED"
    # The FCIDUMP constant is 0; the nuclear repulsion is added here, once.
    assert result.energy + p.mol.energy_nuc() == pytest.approx(p.e_fci, abs=1e-8)
    assert result.carryover_bitstrings_b is not None


def test_uhf_driver_occupancies_match_the_fci_density(tmp_path):
    # NH2 has a non-degenerate ground state, so the occupancies are well defined.
    p = uhf_problem("nh2_doublet")
    prep_unrestricted(tmp_path, p)
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), integrals="unrestricted", **TIGHT)
    run = run_driver(
        UHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = solver_job._read_files(
        work_dir=tmp_path, norb=p.norb, separate_beta=True, integrals="unrestricted"
    )
    fs = direct_uhf.FCISolver()
    fs.conv_tol = 1e-12
    _, civec = fs.kernel((p.h1_a, p.h1_b), (p.h2_aa, p.h2_ab, p.h2_bb), p.norb, p.nelec)
    dm_a, dm_b = fs.make_rdm1s(civec, p.norb, p.nelec)
    np.testing.assert_allclose(result.orbital_occupancies[0], np.diag(dm_a), atol=1e-6)
    np.testing.assert_allclose(result.orbital_occupancies[1], np.diag(dm_b), atol=1e-6)


def test_uhf_driver_in_a_subspace_is_variational_and_above_fci(tmp_path):
    # A product subspace with the Hartree-Fock string first in each spin: the energy lies
    # between FCI and the UHF reference, which the subspace contains.
    p = uhf_problem("nh2_doublet")
    alpha = all_strings(p.norb, p.nelec[0])[:10]
    beta = all_strings(p.norb, p.nelec[1])[:12]
    prep_unrestricted(tmp_path, p, alpha, beta)
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), integrals="unrestricted", **TIGHT)
    run = run_driver(
        UHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert run.returncode == 0, run.stdout + run.stderr
    energy = (
        solver_job._read_files(
            work_dir=tmp_path, norb=p.norb, separate_beta=True, integrals="unrestricted"
        ).energy
        + p.mol.energy_nuc()
    )
    assert p.e_fci - 1e-10 <= energy <= p.mf.e_tot + 1e-10


def restricted_fcidump(work_dir, p):
    solver_job._prep_files(
        work_dir=work_dir,
        ci_strings=(all_strings(p.norb, p.nelec[0]), all_strings(p.norb, p.nelec[1])),
        one_body_tensor=p.h1_a,
        two_body_tensor=p.h2_aa,
        norb=p.norb,
        nelec=p.nelec,
    )


def assert_refused(run, message):
    assert run.returncode != 0, run.stdout
    assert f"[qcsc] ERROR: {message}" in run.stderr
    assert "Davidson energy" not in run.stdout


def test_uhf_driver_refuses_a_restricted_fcidump(tmp_path):
    # Without the check this runs to completion with a wrong Hamiltonian.
    p = uhf_problem("nh2_doublet")
    restricted_fcidump(tmp_path, p)
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path))
    run = run_driver(
        UHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert_refused(run, "the FCIDUMP layout is RESTRICTED (no QCSC_INTEGRALS key)")
    assert not (tmp_path / "davidson_energy.txt").exists()


def test_uhf_driver_refuses_an_unmarked_interleaved_fcidump(tmp_path):
    # Files written before the marker (the source writer) are not accepted by layout guess.
    p = uhf_problem("nh2_doublet")
    prep_unrestricted(tmp_path, p)
    path = tmp_path / "fcidump.txt"
    path.write_text(path.read_text().replace("QCSC_INTEGRALS=SBD_UHF_INTERLEAVED,", "", 1))
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), integrals="unrestricted")
    run = run_driver(
        UHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert_refused(run, "the FCIDUMP layout is RESTRICTED (no QCSC_INTEGRALS key)")


def test_restricted_driver_refuses_an_unrestricted_fcidump(tmp_path):
    p = uhf_problem("nh2_doublet")
    prep_unrestricted(tmp_path, p)
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), integrals="unrestricted")
    run = run_driver(
        RHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert_refused(
        run, "the FCIDUMP layout is SBD_UHF_INTERLEAVED but this binary reads RESTRICTED"
    )


def add_iuhf(work_dir, value):
    path = work_dir / "fcidump.txt"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].rstrip() + f"IUHF={value},"
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("value", ["1", "-1", "yes"])
@pytest.mark.parametrize("binary", ["restricted", "unrestricted"])
def test_both_drivers_refuse_a_molpro_iuhf_fcidump(tmp_path, binary, value):
    p = uhf_problem("nh2_doublet")
    restricted_fcidump(tmp_path, p)
    add_iuhf(tmp_path, value)
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path))
    run = run_driver(
        RHF_BINARY if binary == "restricted" else UHF_BINARY,
        tmp_path,
        solver_job._build_solver_args(solver, separate_beta=True),
    )
    assert_refused(run, f"the FCIDUMP has IUHF={value} (MOLPRO-style block UHF)")


def test_restricted_driver_reads_a_restricted_fcidump_with_iuhf_0(tmp_path):
    # IUHF=0 is MOLPRO's restricted flag: the run is the same as without the key.
    p = uhf_problem("nh2_doublet")
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), **TIGHT)
    args = solver_job._build_solver_args(solver, separate_beta=True)
    energies = {}
    for value in (None, "0"):
        work_dir = tmp_path / f"iuhf_{value}"
        work_dir.mkdir()
        restricted_fcidump(work_dir, p)
        if value is not None:
            add_iuhf(work_dir, value)
        run = run_driver(RHF_BINARY, work_dir, args)
        assert run.returncode == 0, run.stdout + run.stderr
        result = solver_job._read_files(work_dir=work_dir, norb=p.norb, separate_beta=True)
        assert result.integrals == "RESTRICTED"
        energies[value] = result.energy
    # Not byte for byte: a GPU build with several threads gives different last digits from run
    # to run (1e-14 on ROQUO). A different Hamiltonian would move the energy far more than 1e-10.
    assert energies["0"] == pytest.approx(energies[None], abs=1e-10)


def test_uhf_driver_still_refuses_a_restricted_fcidump_with_iuhf_0(tmp_path):
    # IUHF=0 does not declare a layout; the file is restricted.
    p = uhf_problem("nh2_doublet")
    restricted_fcidump(tmp_path, p)
    add_iuhf(tmp_path, "0")
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path))
    run = run_driver(
        UHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert_refused(run, "the FCIDUMP layout is RESTRICTED (no QCSC_INTEGRALS key)")


def test_restricted_driver_still_reads_an_unmarked_restricted_fcidump(tmp_path):
    # FCIDUMPs written before the check have no marker and are restricted.
    p = uhf_problem("nh2_doublet")
    restricted_fcidump(tmp_path, p)
    assert "QCSC_INTEGRALS" not in (tmp_path / "fcidump.txt").read_text()
    solver = solver_job.SBDSolverJob(root_dir=str(tmp_path), **TIGHT)
    run = run_driver(
        RHF_BINARY, tmp_path, solver_job._build_solver_args(solver, separate_beta=True)
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = solver_job._read_files(work_dir=tmp_path, norb=p.norb, separate_beta=True)
    assert result.integrals == "RESTRICTED"
