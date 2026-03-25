from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from qcsc_prefect_dice import io_utils


def test_prep_dice_input_files_writes_expected_inputs(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_from_integrals(path, one_body_tensor, two_body_tensor, norb, nelec):
        captured["path"] = path
        captured["norb"] = norb
        captured["nelec"] = nelec
        Path(path).write_text("FCIDUMP", encoding="utf-8")

    monkeypatch.setattr(io_utils.fcidump, "from_integrals", fake_from_integrals)

    io_utils.prep_dice_input_files(
        work_dir=tmp_path,
        ci_strings=(np.array([1, 3]), np.array([2])),
        one_body_tensor=np.eye(2),
        two_body_tensor=np.zeros((2, 2, 2, 2)),
        norb=2,
        nelec=(1, 1),
        spin_sq=0.0,
        select_cutoff=1e-4,
        davidson_tol=1e-5,
        energy_tol=1e-8,
        max_iter=7,
    )

    assert captured["path"] == str(tmp_path / "fcidump.txt")
    assert captured["norb"] == 2
    assert captured["nelec"] == (1, 1)
    assert (tmp_path / "fcidump.txt").read_text(encoding="utf-8") == "FCIDUMP"
    assert "davidsonTol 1e-05" in (tmp_path / "input.dat").read_text(encoding="utf-8")
    assert (tmp_path / "AlphaDets.bin").stat().st_size == 2 * io_utils.DETERMINANT_BYTES
    assert (tmp_path / "BetaDets.bin").stat().st_size == io_utils.DETERMINANT_BYTES


def test_read_dice_output_files_without_sci_state(tmp_path: Path):
    (tmp_path / "spin1RDM.0.0.txt").write_text(
        "i j val\n0 0 0.8\n1 1 0.1\n2 2 0.2\n3 3 0.9\n",
        encoding="utf-8",
    )
    with (tmp_path / "shci.e").open("wb") as fp:
        fp.write(struct.pack("d", -1.2345))

    result = io_utils.read_dice_output_files(
        work_dir=tmp_path,
        norb=2,
        nelec=(1, 1),
        return_sci_state=False,
    )

    assert result.energy == -1.2345
    np.testing.assert_allclose(result.orbital_occupancies[0], np.array([0.8, 0.2]))
    np.testing.assert_allclose(result.orbital_occupancies[1], np.array([0.1, 0.9]))
    assert result.sci_state is None


def test_read_dice_output_files_with_sci_state(tmp_path: Path):
    (tmp_path / "spin1RDM.0.0.txt").write_text(
        "i j val\n0 0 1.0\n1 1 1.0\n2 2 1.0\n3 3 0.0\n",
        encoding="utf-8",
    )
    with (tmp_path / "shci.e").open("wb") as fp:
        fp.write(struct.pack("d", -2.5))
    with (tmp_path / "dets.bin").open("wb") as fp:
        fp.write(struct.pack("i", 1))
        fp.write(struct.pack("i", 2))
        fp.write(struct.pack("d", 0.75))
        fp.write(b"2a")

    result = io_utils.read_dice_output_files(
        work_dir=tmp_path,
        norb=2,
        nelec=(2, 1),
        return_sci_state=True,
    )

    assert result.energy == -2.5
    assert result.sci_state is not None
    np.testing.assert_allclose(result.sci_state.amplitudes, np.array([[0.75]]))
    np.testing.assert_array_equal(result.sci_state.ci_strs_a, np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(result.sci_state.ci_strs_b, np.array([1], dtype=np.int64))
