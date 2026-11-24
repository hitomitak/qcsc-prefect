"""The UHF reference from an FCIDUMP (qcsc_workflow_utility.uhf).

TestUHFMeanField and TestVariationalOrdering are ported from kunalroy332/qcsc-prefect (9226c1b,
at 3968e58) with only the module changed. The source's ElectronicProperties tests ran the
geometry entry point; here the same checks run on FCIDUMPs written from the same molecules.

Ground-truth energies and spins come from independent PySCF RHF/UHF/FCI calculations (sto-3g),
so the assertions are anchored to physics, not to the code under test.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import qcsc_workflow_utility.uhf as uhf
from pyscf import fci, gto, scf
from pyscf.tools import fcidump

H4_STRETCHED = "H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0"
H4_EQUILIBRIUM = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
H4_TRIPLET = "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"
H5_DOUBLET = "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0; H 0 0 4.0"
NH2_DOUBLET = "N 0 0 0; H 0 0.80 0.59; H 0 -0.80 0.59"


def _rhf(atom, spin=0):
    return scf.RHF(gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)).run(verbose=0)


def _fci_energy(atom, spin):
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    return float(fci.FCI(mf).kernel()[0])


def _write_fcidump(path, atom, spin=0):
    """Restricted FCIDUMP of all orbitals, from (RO)HF orbitals; ECORE = E_nuc."""
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    fcidump.from_scf(mf, str(path))
    return str(path)


def _reference(tmp_path, atom, spin=0, **kwargs):
    kwargs.setdefault("cache_dir", None)
    path = _write_fcidump(tmp_path / "fcidump.txt", atom, spin)
    return uhf.compute_uhf_integrals_from_fcidump.fn(path, **kwargs)


def _energy_from_integrals(ep):
    """UHF determinant energy from the returned integrals (occupied = first n orbitals)."""
    na, nb = ep.num_electrons
    a, b = slice(0, na), slice(0, nb)
    gaa, gab, gbb = ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb
    return float(
        np.trace(ep.one_body_tensor[a, a])
        + np.trace(ep.one_body_tensor_b[b, b])
        + 0.5 * (np.einsum("iijj", gaa[a, a, a, a]) - np.einsum("ijji", gaa[a, a, a, a]))
        + 0.5 * (np.einsum("iijj", gbb[b, b, b, b]) - np.einsum("ijji", gbb[b, b, b, b]))
        + np.einsum("iijj", gab[a, a, b, b])
        + ep.nuclear_repulsion_energy
    )


def _fields(ep):
    return {k: np.asarray(v) for k, v in ep.model_dump().items() if isinstance(v, np.ndarray)} | {
        "occ_a": np.asarray(ep.initial_occupancy[0]),
        "occ_b": np.asarray(ep.initial_occupancy[1]),
    }


def _assert_bitwise_equal(ep1, ep2):
    f1, f2 = _fields(ep1), _fields(ep2)
    assert f1.keys() == f2.keys()
    for k in f1:
        assert f1[k].tobytes() == f2[k].tobytes(), k
    assert ep1.reference_energy == ep2.reference_energy


# ── mean-field level: _converge_broken_symmetry_uhf (ported from 9226c1b) ───────────────────


class TestUHFMeanField:
    def test_stretched_singlet_breaks_symmetry(self):
        """Strongly-correlated H4 singlet: UHF must break spin symmetry and drop well below RHF."""
        mol = gto.M(atom=H4_STRETCHED, basis="sto-3g", spin=0, verbose=0)
        e_rhf = _rhf(H4_STRETCHED).e_tot
        mf = uhf._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert ss > 0.5, f"UHF stayed restricted (<S^2>={ss:.3f}) on a correlated singlet"
        assert mf.e_tot < e_rhf - 1e-2, f"UHF {mf.e_tot:.6f} not below RHF {e_rhf:.6f}"
        assert mf.e_tot >= _fci_energy(H4_STRETCHED, 0) - 1e-8, "UHF below FCI (non-variational!)"

    def test_equilibrium_singlet_stays_restricted(self):
        """Weakly-correlated H4 singlet: NO spin instability exists, so UHF must equal RHF
        (<S^2>=0). The fix must not spuriously break symmetry where it shouldn't."""
        mol = gto.M(atom=H4_EQUILIBRIUM, basis="sto-3g", spin=0, verbose=0)
        e_rhf = _rhf(H4_EQUILIBRIUM).e_tot
        mf = uhf._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert ss < 1e-3, f"UHF spuriously broke symmetry (<S^2>={ss:.3e}) at equilibrium"
        assert abs(mf.e_tot - e_rhf) < 1e-6, f"UHF {mf.e_tot:.8f} != RHF {e_rhf:.8f} at equilibrium"

    def test_triplet_spin(self):
        """H4 triplet (2Sz=2): <S^2> should be ~2 (S=1 -> S(S+1)=2) and energy >= FCI."""
        mol = gto.M(atom=H4_TRIPLET, basis="sto-3g", spin=2, verbose=0)
        mf = uhf._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert abs(ss - 2.0) < 0.15, f"triplet <S^2>={ss:.3f} not ~2.0"
        assert mf.e_tot >= _fci_energy(H4_TRIPLET, 2) - 1e-8, "triplet UHF below FCI"

    def test_doublet_spin(self):
        """H5 doublet (2Sz=1): <S^2> near 0.75 (+ some contamination), energy >= FCI."""
        mol = gto.M(atom=H5_DOUBLET, basis="sto-3g", spin=1, verbose=0)
        mf = uhf._converge_broken_symmetry_uhf(scf.UHF(mol))
        ss, _ = mf.spin_square()
        assert 0.7 <= ss < 1.5, f"doublet <S^2>={ss:.3f} out of expected range"
        assert mf.e_tot >= _fci_energy(H5_DOUBLET, 1) - 1e-8, "doublet UHF below FCI"


class TestVariationalOrdering:
    @pytest.mark.parametrize("atom,spin", [(H4_STRETCHED, 0), (H4_EQUILIBRIUM, 0)])
    def test_uhf_le_rhf_le_none_via_pipeline(self, atom, spin):
        """The pipeline's UHF mean-field energy must be <= its RHF mean-field energy and >= FCI.
        (Uses the mean-field energies the pipeline actually converges, ensuring the reference the
        SQD seed is built from respects the variational ordering and is not restricted-in-disguise
        for the strongly-correlated case.)"""
        e_rhf = _rhf(atom, spin).e_tot
        mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
        mf_u = uhf._converge_broken_symmetry_uhf(scf.UHF(mol))
        e_fci = _fci_energy(atom, spin)
        assert mf_u.e_tot <= e_rhf + 1e-9, f"UHF {mf_u.e_tot:.6f} > RHF {e_rhf:.6f}"
        assert mf_u.e_tot >= e_fci - 1e-8, f"UHF {mf_u.e_tot:.6f} < FCI {e_fci:.6f}"


# ── FCIDUMP entry point: compute_uhf_integrals_from_fcidump ─────────────────────────────────


class TestFromFcidump:
    def test_blocks_shapes_and_electron_numbers(self, tmp_path):
        ep = _reference(tmp_path, H4_TRIPLET, spin=2)
        norb, (na, nb) = ep.num_orbitals, ep.num_electrons
        assert (norb, na, nb) == (4, 3, 1)
        assert ep.unrestricted is True and ep.open_shell is True
        for blk in (ep.one_body_tensor, ep.one_body_tensor_b):
            assert blk.shape == (norb, norb)
        for blk in (ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb):
            assert blk.shape == (norb,) * 4
        assert ep.t2.shape == (na, na, norb - na, norb - na)
        assert ep.t2_ab.shape == (na, nb, norb - na, norb - nb)
        assert ep.t2_bb.shape == (nb, nb, norb - nb, norb - nb)
        assert [len(o) for o in ep.initial_occupancy] == [norb, norb]
        assert (ep.reference_tag, ep.reference_source, ep.fcidump_sha256) == (None, None, None)

    @pytest.mark.parametrize(("atom", "spin"), [(H4_STRETCHED, 0), (NH2_DOUBLET, 1)])
    def test_integrals_reproduce_the_reference_energy(self, tmp_path, atom, spin):
        ep = _reference(tmp_path, atom, spin)
        assert abs(_energy_from_integrals(ep) - ep.reference_energy) < 1e-10

    def test_integrals_have_the_permutation_symmetry_of_real_orbitals(self, tmp_path):
        ep = _reference(tmp_path, H4_STRETCHED)
        for h1 in (ep.one_body_tensor, ep.one_body_tensor_b):
            np.testing.assert_allclose(h1, h1.T, atol=1e-12)
        for g in (ep.two_body_tensor, ep.two_body_tensor_bb):
            np.testing.assert_allclose(g, g.transpose(1, 0, 2, 3), atol=1e-12)
            np.testing.assert_allclose(g, g.transpose(2, 3, 0, 1), atol=1e-12)
        g = ep.two_body_tensor_ab
        np.testing.assert_allclose(g, g.transpose(1, 0, 2, 3), atol=1e-12)
        np.testing.assert_allclose(g, g.transpose(0, 1, 3, 2), atol=1e-12)

    def test_matches_an_independent_uhf_of_the_molecule(self, tmp_path):
        # With every orbital active, the UHF minimum does not depend on the orbitals the
        # FCIDUMP was written in.
        mol = gto.M(atom=NH2_DOUBLET, basis="sto-3g", spin=1, verbose=0)
        ref = scf.UHF(mol).run(conv_tol=1e-12)
        ep = _reference(tmp_path, NH2_DOUBLET, spin=1)
        assert abs(ep.reference_energy - ref.e_tot) < 1e-8
        assert abs(ep.reference_spin_square - ref.spin_square()[0]) < 1e-6
        assert ep.nuclear_repulsion_energy == pytest.approx(mol.energy_nuc(), abs=1e-12)

    def test_stretched_singlet_is_spin_broken_and_variational(self, tmp_path):
        ep = _reference(tmp_path, H4_STRETCHED)
        assert ep.reference_spin_square > 0.5
        assert _fci_energy(H4_STRETCHED, 0) - 1e-8 <= ep.reference_energy
        assert ep.reference_energy < _rhf(H4_STRETCHED).e_tot - 1e-2

    def test_occupancies_are_fractional_and_sum_to_the_electron_numbers(self, tmp_path):
        """The occupancy seed is the diagonal of the correlated UCCSD 1-RDM: fractional for a
        correlated system (integer 0/1 occupancies give recovery no signal)."""
        ep = _reference(tmp_path, H4_STRETCHED)
        occ_a, occ_b = (np.asarray(o) for o in ep.initial_occupancy)
        frac = np.sum((occ_a > 0.01) & (occ_a < 0.99)) + np.sum((occ_b > 0.01) & (occ_b < 0.99))
        assert frac > 0
        assert occ_a.sum() == pytest.approx(ep.num_electrons[0], abs=1e-8)
        assert occ_b.sum() == pytest.approx(ep.num_electrons[1], abs=1e-8)

    def test_ccsd_settings_are_used(self, tmp_path, monkeypatch):
        seen = []
        real = uhf.cc.UCCSD

        def recording_uccsd(mf):
            mycc = real(mf)
            kernel = mycc.kernel

            def counted(*args, **kwargs):
                seen.append(mycc.max_cycle)
                return kernel(*args, **kwargs)

            mycc.kernel = counted
            return mycc

        monkeypatch.setattr(uhf.cc, "UCCSD", recording_uccsd)
        _reference(tmp_path, H4_STRETCHED, ccsd_max_cycle=1, ccsd_restarts=3)
        assert seen == [1, 1, 1]  # not converged in one cycle: three runs of one cycle


class TestCache:
    def test_key_and_file_name(self, tmp_path):
        path = _write_fcidump(tmp_path / "fcidump.txt", NH2_DOUBLET, spin=1)
        cache = tmp_path / "cache"
        ep = uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=str(cache))
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        key = f"v2|{sha}|af=None|nelec=(5, 4)"
        assert ep.reference_cache_key == key
        name = hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]
        assert [p.name for p in cache.iterdir()] == [f"uhf_bs_{name}.npz"]
        with np.load(cache / f"uhf_bs_{name}.npz") as c:
            assert sorted(c.files) == sorted(
                ["mo_a", "mo_b", "occ_a", "occ_b", "ene_a", "ene_b", "e_tot", "nelec"]
            )
            assert float(c["e_tot"]) == ep.reference_energy

    def test_a_cached_reference_is_reused_bitwise(self, tmp_path, monkeypatch):
        path = _write_fcidump(tmp_path / "fcidump.txt", H4_STRETCHED)
        cache = str(tmp_path / "cache")
        first = uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=cache)

        def no_scf(*args, **kwargs):
            raise AssertionError("the cached reference was not reused")

        monkeypatch.setattr(uhf, "_converge_broken_symmetry_uhf", no_scf)
        again = uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=cache)
        _assert_bitwise_equal(first, again)

    def test_cache_read_false_recomputes_and_rewrites(self, tmp_path, monkeypatch):
        path = _write_fcidump(tmp_path / "fcidump.txt", H4_STRETCHED)
        cache = tmp_path / "cache"
        uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=str(cache))
        (cached,) = cache.iterdir()
        cached.write_bytes(b"stale")
        calls = []
        real = uhf._converge_broken_symmetry_uhf
        monkeypatch.setattr(
            uhf, "_converge_broken_symmetry_uhf", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=str(cache), cache_read=False)
        assert calls == [1]
        with np.load(cached) as c:
            assert "mo_a" in c.files

    def test_a_cache_for_other_electron_numbers_is_ignored(self, tmp_path, monkeypatch):
        path = _write_fcidump(tmp_path / "fcidump.txt", H4_STRETCHED)
        cache = tmp_path / "cache"
        uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=str(cache))
        (cached,) = cache.iterdir()
        with np.load(cached) as c:
            data = {k: c[k] for k in c.files}
        data["nelec"] = np.array([3, 1])
        np.savez(cached, **data)
        calls = []
        real = uhf._converge_broken_symmetry_uhf
        monkeypatch.setattr(
            uhf, "_converge_broken_symmetry_uhf", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=str(cache))
        assert calls == [1]

    def test_cache_dir_none_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ep = _reference(tmp_path, H4_STRETCHED, cache_dir=None)
        assert ep.reference_cache_key.startswith("v2|")
        assert not (tmp_path / "home").exists()


class TestAFGroups:
    def test_spec_always_has_the_polarization(self):
        assert uhf._af_guess_spec(None) is None
        spec = uhf._af_guess_spec({"a": [0], "b": [1], "up": ["a"], "down": ["b"]})
        assert spec == {"a": [0], "b": [1], "up": ["a"], "down": ["b"], "pol": 1.0}
        assert uhf._af_guess_spec({"a": [0], "pol": 0.6})["pol"] == 0.6
        with pytest.raises(ValueError, match="must be a mapping"):
            uhf._af_guess_spec("fe4s4")

    def test_guess_density_fill(self):
        spec = {"a": [0], "b": [1], "s": [2], "c": [3], "up": ["a"], "down": ["b"]}
        dm = uhf._density_from_af_groups({**spec, "free": ["s"], "pol": 0.6}, 5)
        np.testing.assert_allclose(np.diag(dm[0]), [0.8, 0.2, 0.5, 1.0, 0.0])
        np.testing.assert_allclose(np.diag(dm[1]), [0.2, 0.8, 0.5, 1.0, 0.0])

    def test_out_of_range_orbitals_fail_before_any_scf(self, tmp_path, monkeypatch):
        def no_scf(*args, **kwargs):
            raise AssertionError("SCF ran")

        monkeypatch.setattr(uhf, "_converge_broken_symmetry_uhf", no_scf)
        with pytest.raises(ValueError, match="out of range for NORB=4"):
            _reference(tmp_path, H4_STRETCHED, af_groups={"a": [0, 7], "up": ["a"]})

    def test_af_groups_enter_the_key_and_never_raise_the_energy(self, tmp_path):
        groups = {"a": [0, 1], "b": [2, 3], "up": ["a"], "down": ["b"]}
        plain = _reference(tmp_path, H4_STRETCHED)
        af = _reference(tmp_path, H4_STRETCHED, af_groups=groups)
        token = "af=" + str({**groups, "pol": 1.0})
        assert af.reference_cache_key.split("|")[2] == token
        # The AF solution replaces the default one only when it is lower.
        assert af.reference_energy <= plain.reference_energy + 1e-6


def test_environment_variables_of_the_source_are_not_read(tmp_path, monkeypatch):
    path = _write_fcidump(tmp_path / "fcidump.txt", H4_STRETCHED)
    plain = uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=None)
    for name, value in {
        "FE4S4_AF_GROUPS": "fe4s4",
        "FE4S4_AF_POL": "0.5",
        "FE4S4_AF_FREE_S": "1",
        "SBD_UHF_CACHE": "0",
        "SBD_UHF_CACHE_DIR": str(tmp_path / "env-cache"),
        "SEED_CCSD_MAX_CYCLE": "1",
        "SEED_CCSD_RESTARTS": "1",
        "OO_INIT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    with_env = uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=None)
    _assert_bitwise_equal(plain, with_env)
    assert with_env.reference_cache_key == plain.reference_cache_key
    assert not (tmp_path / "env-cache").exists()


def _add_header_key(path, token, line=0):
    """Append ``token`` (for example ``"IUHF=1,"``) to header line ``line`` of an FCIDUMP."""
    lines = open(path).read().splitlines()
    lines[line] = lines[line].rstrip() + token
    open(path, "w").write("\n".join(lines) + "\n")


class TestInputLayout:
    @pytest.mark.parametrize(
        ("token", "line", "message"),
        [
            ("IUHF=1,", 0, "IUHF=1 is a MOLPRO-style UHF FCIDUMP"),
            ("IUHF=-1,", 0, "IUHF=-1 is a MOLPRO-style UHF FCIDUMP"),
            ("IUHF=YES,", 0, "IUHF=YES is a MOLPRO-style UHF FCIDUMP"),
            ("IUHF=1,", 2, "IUHF=1 is a MOLPRO-style UHF FCIDUMP"),  # on the ISYM line
            ("QCSC_INTEGRALS=SBD_UHF_INTERLEAVED,", 0, "QCSC_INTEGRALS=SBD_UHF_INTERLEAVED"),
        ],
    )
    def test_uhf_layouts_are_refused_before_the_integrals_are_read(
        self, tmp_path, monkeypatch, token, line, message
    ):
        # Read as a restricted file, these give a different Hamiltonian without an error.
        path = _write_fcidump(tmp_path / "fcidump.txt", NH2_DOUBLET, spin=1)
        _add_header_key(path, token, line)

        def not_reached(*args, **kwargs):
            raise AssertionError("the integrals were read")

        monkeypatch.setattr(uhf.tools.fcidump, "read", not_reached)
        monkeypatch.setattr(uhf.tools.fcidump, "to_scf", not_reached)
        with pytest.raises(ValueError, match=message):
            uhf.compute_uhf_integrals_from_fcidump.fn(path, cache_dir=None)

    def test_an_sbd_interleaved_fcidump_is_refused(self, tmp_path):
        # The layout qcsc_prefect_sbd writes for its -D_UHF driver: spin-orbital indices up to
        # 2 * NORB. PySCF's restricted reader would fail on them, or misread them if in range.
        path = tmp_path / "fcidump.txt"
        path.write_text(
            " &FCI NORB=2,NELEC=2,MS2=0,QCSC_INTEGRALS=SBD_UHF_INTERLEAVED,\n"
            "  ORBSYM=1,1,\n  ISYM=1,\n &END\n"
            "  5.0000000000000000e-01 1 1 1 1\n  4.0000000000000000e-01 4 4 4 4\n"
            " -1.0000000000000000e+00 1 1 0 0\n -9.0000000000000000e-01 4 4 0 0\n"
            "  0.0000000000000000e+00 0 0 0 0\n"
        )
        with pytest.raises(ValueError, match="is not a restricted FCIDUMP"):
            uhf.compute_uhf_integrals_from_fcidump.fn(str(path), cache_dir=None)

    @pytest.mark.parametrize("token", ["IUHF=0,", "QCSC_INTEGRALS=RESTRICTED,"])
    def test_restricted_flags_change_nothing(self, tmp_path, token):
        plain = _write_fcidump(tmp_path / "plain.txt", NH2_DOUBLET, spin=1)
        flagged = _write_fcidump(tmp_path / "flagged.txt", NH2_DOUBLET, spin=1)
        _add_header_key(flagged, token)
        _assert_bitwise_equal(
            uhf.compute_uhf_integrals_from_fcidump.fn(plain, cache_dir=None),
            uhf.compute_uhf_integrals_from_fcidump.fn(flagged, cache_dir=None),
        )


@pytest.fixture(scope="module")
def prefect_api():
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


def test_a_changed_fcidump_at_the_same_path_is_recomputed_in_one_flow_run(prefect_api, tmp_path):
    from prefect import flow

    path = _write_fcidump(tmp_path / "fcidump.txt", NH2_DOUBLET, spin=1)

    def shift_ecore(delta):
        lines = open(path).read().splitlines()
        value, *indices = lines[-1].split()
        assert indices == ["0", "0", "0", "0"]
        lines[-1] = f" {float(value) + delta:.16e}  0  0  0  0"
        open(path, "w").write("\n".join(lines) + "\n")

    @flow
    def twice():
        first = uhf.compute_uhf_integrals_from_fcidump(path, cache_dir=None, return_state=True)
        shift_ecore(1.5)
        second = uhf.compute_uhf_integrals_from_fcidump(path, cache_dir=None, return_state=True)
        return first, second

    first, second = twice()
    assert (first.name, second.name) == ("Completed", "Completed")  # neither is "Cached"
    e_first = first.result().nuclear_repulsion_energy
    e_second = second.result().nuclear_repulsion_energy
    assert e_second == pytest.approx(e_first + 1.5, abs=1e-12)


def _fake_reference():
    return SimpleNamespace(
        mo_coeff=(np.eye(2), np.eye(2)),
        mo_occ=(np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        mo_energy=(np.array([-1.0, 1.0]), np.array([-0.9, 1.1])),
        e_tot=-1.5,
        mol=SimpleNamespace(nelec=(1, 1)),
    )


class TestCacheWrites:
    def test_concurrent_writers_of_one_reference_do_not_share_a_temporary_file(
        self, tmp_path, monkeypatch, caplog
    ):
        cpath = str(tmp_path / "cache" / "uhf_bs_0123456789abcdef.npz")
        both_open = threading.Barrier(2, timeout=30)
        names = []
        real_savez = np.savez

        def savez_when_both_writers_are_open(f, **arrays):
            names.append(f.name)
            both_open.wait()
            real_savez(f, **arrays)

        monkeypatch.setattr(uhf.np, "savez", savez_when_both_writers_are_open)
        writers = [
            threading.Thread(target=uhf._uhf_cache_save, args=(cpath, _fake_reference()))
            for _ in range(2)
        ]
        with caplog.at_level(logging.WARNING, logger=uhf.__name__):
            for w in writers:
                w.start()
            for w in writers:
                w.join(timeout=60)
        assert len(set(names)) == 2, names
        assert all(n.startswith(cpath + ".") and n.endswith(".tmp") for n in names)
        assert not [r for r in caplog.records if "save failed" in r.getMessage()]
        assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [
            "uhf_bs_0123456789abcdef.npz"
        ]
        with np.load(cpath) as c:
            assert float(c["e_tot"]) == -1.5
            assert c["nelec"].tolist() == [1, 1]

    def test_a_failed_write_leaves_no_temporary_file(self, tmp_path, monkeypatch, caplog):
        cpath = str(tmp_path / "cache" / "uhf_bs_0123456789abcdef.npz")

        def failing_savez(f, **arrays):
            f.write(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(uhf.np, "savez", failing_savez)
        with caplog.at_level(logging.WARNING, logger=uhf.__name__):
            uhf._uhf_cache_save(cpath, _fake_reference())
        assert any("save failed" in r.getMessage() for r in caplog.records)
        assert list((tmp_path / "cache").iterdir()) == []
