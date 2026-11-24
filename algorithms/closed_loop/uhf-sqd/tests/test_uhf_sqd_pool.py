"""The device path of walker_sqd: reset mitigation and the saved sample pool.

The pool is the source's (kunalroy332/qcsc-prefect at 3968e58: b9023b1, d3d97b0, 3968e58): a
compressed npz with packed_bits, num_bits, num_shots and a JSON ref_meta, written before any
diagonalization. ref_meta has the source's five keys with fcidump_sha256 filled, plus
reference_cache_key and bit_layout.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays
from uhf_sqd import data_io, sqd
from uhf_sqd_helpers import KEY, NA, NB, NORB, FakeSolver, fake_props, prepare_walker, run_walker

SOURCE_KEYS = {"norb", "nelec", "fcidump_sha256", "reference_tag", "method"}


class DeviceSource:
    """A QuantumSource that needs a circuit and returns fixed meas and test registers."""

    backend_name = "fake-device"
    requires_circuit = True

    def __init__(self, meas: np.ndarray, test: np.ndarray):
        self.meas, self.test = meas, test
        self.circuits = []

    def sample(self, circuit=None, *, shots):
        self.circuits.append(circuit)
        data = DataBin(
            meas=BitArray.from_bool_array(self.meas), test=BitArray.from_bool_array(self.test)
        )
        return PrimitiveResult([SamplerPubResult(data, metadata={})], metadata={})


def device_samples(n=300, n_rejected=60, seed=0):
    rng = np.random.default_rng(seed)
    meas = rng.integers(2, size=(n, 2 * NORB), dtype=bool)
    test = np.zeros((n, 2 * NORB), dtype=bool)
    test[rng.choice(n, n_rejected, replace=False), 0] = True
    return meas, test


@pytest.fixture
def storage(monkeypatch, tmp_path):
    monkeypatch.setattr(data_io, "S3Bucket", None)
    monkeypatch.setenv("SBD_LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(sqd, "create_lucj_circuit", lambda **kwargs: "lucj-circuit")
    return tmp_path


def _pool(telemetry):
    path = telemetry["raw_samples_path"].removeprefix("file://")
    with np.load(path, allow_pickle=False) as z:
        return path, {k: z[k] for k in z.files}


def test_the_pool_keeps_the_mitigated_shots_with_their_multiplicities(storage, monkeypatch):
    meas, test = device_samples()
    source = DeviceSource(meas, test)
    prepare_walker(monkeypatch, solver=FakeSolver(), source=source)
    _, telemetry = run_walker()
    assert source.circuits == ["lucj-circuit"]
    kept = ~test.any(axis=1)
    assert telemetry["shot_retention_rate"] == pytest.approx(kept.mean())
    path, pool = _pool(telemetry)
    assert path.startswith(str(storage)) and "raw_samples_t00_w0_" in path
    assert sorted(pool) == ["num_bits", "num_shots", "packed_bits", "ref_meta"]
    assert pool["packed_bits"].dtype == np.uint8
    assert pool["num_bits"].dtype == np.int64 and pool["num_bits"].tolist() == [2 * NORB]
    assert pool["num_shots"].tolist() == [int(kept.sum())]
    reloaded = BitArray(pool["packed_bits"], int(pool["num_bits"][0]))
    expected = BitArray.from_samples(
        BitArray.from_bool_array(meas).get_bitstrings(kept), num_bits=2 * NORB
    )
    assert reloaded.num_shots == expected.num_shots
    assert reloaded.array.tobytes() == expected.array.tobytes()


def test_recovery_on_the_reloaded_pool_is_the_same(storage, monkeypatch):
    meas, test = device_samples()
    prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(meas, test))
    _, telemetry = run_walker()
    _, pool = _pool(telemetry)
    kept = ~test.any(axis=1)
    original = BitArray.from_samples(
        BitArray.from_bool_array(meas).get_bitstrings(kept), num_bits=2 * NORB
    )
    reloaded = BitArray(pool["packed_bits"], int(pool["num_bits"][0]))
    occ = (np.full(NORB, NA / NORB), np.full(NORB, NB / NORB))
    runs = []
    for bits in (original, reloaded):
        matrix, probs = bit_array_to_arrays(bits)
        runs.append(recover_configurations(matrix, probs, occ, NA, NB, rand_seed=5))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def test_ref_meta_keeps_the_source_keys_and_adds_the_reference_and_bit_layout(storage, monkeypatch):
    prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(*device_samples()))
    _, telemetry = run_walker()
    _, pool = _pool(telemetry)
    assert pool["ref_meta"].dtype == np.dtype("<U2048") and pool["ref_meta"].shape == (1,)
    meta = json.loads(str(pool["ref_meta"][0]))
    assert set(meta) == SOURCE_KEYS | {"reference_cache_key", "bit_layout"}
    assert meta["norb"] == NORB and meta["nelec"] == [NA, NB] and meta["method"] == "uhf"
    assert meta["fcidump_sha256"] == KEY.split("|")[1]
    assert meta["reference_tag"] is None
    assert meta["reference_cache_key"] == KEY
    assert meta["bit_layout"] == sqd.BIT_LAYOUT
    # The telemetry keeps the source's record (fcidump_sha256 None for an FCIDUMP input).
    assert telemetry["fcidump_sha256"] is None


def test_bit_layout_puts_alpha_on_the_right_half(storage, monkeypatch):
    # One shot with only alpha orbital 0 occupied (qubit 0) and beta orbital 1 (qubit norb + 1).
    meas = np.zeros((1, 2 * NORB), dtype=bool)
    bits = np.zeros(2 * NORB, dtype=bool)
    bits[0] = bits[NORB + 1] = True
    # from_bool_array takes rows in printed order (clbit 2norb-1 first).
    meas[0] = bits[::-1]
    matrix, _ = bit_array_to_arrays(BitArray.from_bool_array(meas))
    alpha, beta = matrix[0, NORB:], matrix[0, :NORB]
    assert alpha[NORB - 1 - 0] and alpha.sum() == 1  # orbital p is column norb-1-p
    assert beta[NORB - 1 - 1] and beta.sum() == 1
    assert "alpha=q[0:norb],beta=q[norb:2norb]" in sqd.BIT_LAYOUT


def test_reset_mitigation_that_rejects_every_shot_falls_back_with_a_warning(storage, monkeypatch):
    meas, test = device_samples()
    test[:, 0] = True
    logger = prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(meas, test))
    _, telemetry = run_walker()
    assert any("Reset mitigation kept 0/300" in w for w in logger.warnings)
    _, pool = _pool(telemetry)
    assert pool["num_shots"].tolist() == [300]
    # The source's retention rate is 1.0 here, as when every shot passes; the telemetry says
    # which case it was.
    assert telemetry["shot_retention_rate"] == 1.0
    assert telemetry["reset_mitigation_fallback"] is True
    assert telemetry["reset_test_passed_shots"] == 0


def test_telemetry_records_the_shots_that_passed_the_reset_test(storage, monkeypatch):
    meas, test = device_samples(n_rejected=60)
    prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(meas, test))
    _, telemetry = run_walker()
    assert telemetry["reset_mitigation_fallback"] is False
    assert telemetry["reset_test_passed_shots"] == 240
    assert telemetry["shot_retention_rate"] == pytest.approx(0.8)


def test_without_reset_mitigation_every_shot_is_kept(storage, monkeypatch):
    meas, test = device_samples()
    prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(meas, test))
    _, telemetry = run_walker(use_reset_mitigation=False)
    _, pool = _pool(telemetry)
    assert pool["num_shots"].tolist() == [300] and telemetry["shot_retention_rate"] == 1.0
    assert "reset_mitigation_fallback" not in telemetry


def _af_key(af_groups):
    sha = "cd" * 32
    spec = {**af_groups, "pol": 1.0}
    return sha, f"v2|{sha}|af={spec}|nelec=({NA}, {NB})"


def test_a_long_ref_meta_is_not_truncated(storage, monkeypatch):
    # An AF guess over a 36-orbital active space with descriptive fragment names; each orbital
    # appears in a few overlapping fragments.
    groups = {
        f"iron_sulfur_ligand_fragment_{i:02d}": [i, (i + 1) % 36, (i + 2) % 36] for i in range(36)
    }
    groups |= {"up": [f"iron_sulfur_ligand_fragment_{i:02d}" for i in range(0, 36, 2)]}
    sha, key = _af_key(groups)
    assert len(key) > 2048
    ep = fake_props()
    ep.reference_cache_key = key
    prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(*device_samples()))
    _, telemetry = run_walker(ep=ep)
    _, pool = _pool(telemetry)
    meta = json.loads(str(pool["ref_meta"][0]))  # the whole JSON reads back
    assert meta["reference_cache_key"] == key and meta["fcidump_sha256"] == sha
    assert pool["ref_meta"].shape == (1,) and pool["ref_meta"].dtype.kind == "U"


def test_a_bar_in_a_fragment_name_keeps_the_fcidump_hash(storage, monkeypatch):
    sha, key = _af_key({"fe|1": [0, 1], "up": ["fe|1"]})
    assert key.count("|") > 3
    assert sqd._fcidump_sha256(key) == sha
    ep = fake_props()
    ep.reference_cache_key = key
    prepare_walker(monkeypatch, solver=FakeSolver(), source=DeviceSource(*device_samples()))
    _, telemetry = run_walker(ep=ep)
    _, pool = _pool(telemetry)
    assert json.loads(str(pool["ref_meta"][0]))["fcidump_sha256"] == sha


def test_a_failed_pool_save_does_not_stop_the_walker(storage, monkeypatch):
    real = sqd.save_ndarray

    def save(file_prefix, **arrays):
        if file_prefix.startswith("raw_samples"):
            raise OSError("disk full")
        return real(file_prefix=file_prefix, **arrays)

    solver = FakeSolver()
    prepare_walker(
        monkeypatch, solver=solver, source=DeviceSource(*device_samples()), save_ndarray=save
    )
    (energy, _), telemetry = run_walker()
    assert np.isfinite(energy) and len(solver.calls) == 1
    assert "raw_samples_path" not in telemetry and telemetry["sqd_data"].startswith("file://")


def test_a_random_source_writes_no_pool(storage, monkeypatch):
    prepare_walker(monkeypatch, solver=FakeSolver())
    _, telemetry = run_walker()
    assert "raw_samples_path" not in telemetry
    assert not list(storage.rglob("raw_samples_*"))


@pytest.mark.parametrize(
    "key",
    [
        "v1|" + "ab" * 32 + "|af=None|nelec=(1, 1)",
        "v2|abc|af=None|nelec=(1, 1)",
        "v2|" + "ab" * 32 + "|af=None",
        "v2|" + "ab" * 32,
    ],
)
def test_ref_meta_refuses_an_unexpected_reference_key(key):
    with pytest.raises(ValueError, match="reference_cache_key"):
        sqd._fcidump_sha256(key)
