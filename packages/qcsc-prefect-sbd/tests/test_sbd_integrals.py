"""The unrestricted FCIDUMP writer against an emulation of SBD's -D_UHF reader."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from qcsc_prefect_sbd import integrals as mod
from sbd_uhf_helpers import uhf_tensors


def read_like_sbd_uhf(path: Path, norb: int):
    """Emulate SBD makeintegrals.h `#ifdef _UHF` SetupIntegrals (0-based spin orbitals).

    One-body records set I1(i-1, j-1) and its transpose; two-body records set the eight index
    permutations of (i-1, j-1, k-1, l-1); the record 0 0 0 0 is the constant I0.
    """
    i1 = np.zeros((2 * norb, 2 * norb))
    i2 = np.zeros((2 * norb,) * 4)
    i0 = 0.0
    lines = Path(path).read_text().splitlines()
    body = lines[lines.index(" &END") + 1 :]
    for line in body:
        value, *indices = line.split()
        value = float(value)
        i, j, k, m = (int(x) for x in indices)
        if i == j == k == m == 0:
            i0 = value
        elif k == m == 0:
            i1[i - 1, j - 1] = i1[j - 1, i - 1] = value
        else:
            for p, q, r, s in (
                (i, j, k, m), (j, i, k, m), (i, j, m, k), (j, i, m, k),
                (k, m, i, j), (k, m, j, i), (m, k, i, j), (m, k, j, i),
            ):  # fmt: skip
                i2[p - 1, q - 1, r - 1, s - 1] = value
    return i0, i1, i2


def expected_spin_orbital(t, norb):
    """Spin-orbital integrals with alpha = 2p, beta = 2p + 1 (0-based)."""
    i1 = np.zeros((2 * norb, 2 * norb))
    i2 = np.zeros((2 * norb,) * 4)
    for p, q in itertools.product(range(norb), repeat=2):
        i1[2 * p, 2 * q] = t["h1_a"][p, q]
        i1[2 * p + 1, 2 * q + 1] = t["h1_b"][p, q]
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        i2[2 * p, 2 * q, 2 * r, 2 * s] = t["h2_aa"][p, q, r, s]
        i2[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = t["h2_bb"][p, q, r, s]
        i2[2 * p, 2 * q, 2 * r + 1, 2 * s + 1] = t["h2_ab"][p, q, r, s]
        i2[2 * p + 1, 2 * q + 1, 2 * r, 2 * s] = t["h2_ab"][r, s, p, q]
    return i1, i2


@pytest.mark.parametrize("norb", [1, 3, 5])
def test_writer_reproduces_every_spin_block_through_the_sbd_reader(tmp_path, norb):
    t = uhf_tensors(norb, seed=norb)
    path = tmp_path / "fcidump.txt"
    mod.write_sbd_uhf_fcidump(path, norb=norb, nelec=(1, 1), **t)
    i0, i1, i2 = read_like_sbd_uhf(path, norb)
    e1, e2 = expected_spin_orbital(t, norb)
    assert i0 == 0.0
    # Values are written with 16 digits after the decimal point in exponent form.
    np.testing.assert_allclose(i1, e1, rtol=1e-15, atol=0)
    np.testing.assert_allclose(i2, e2, rtol=1e-15, atol=0)


def test_header_declares_the_layout_on_the_first_line(tmp_path):
    path = tmp_path / "fcidump.txt"
    mod.write_sbd_uhf_fcidump(path, norb=2, nelec=(2, 1), **uhf_tensors(2))
    lines = path.read_text().splitlines()
    # On the first line: SBD joins header lines without a separator, so a key on a later line
    # would only be read if the previous line ended with a comma.
    assert lines[0] == " &FCI NORB=2,NELEC=3,MS2=1,QCSC_INTEGRALS=SBD_UHF_INTERLEAVED,"
    assert lines[1:4] == ["  ORBSYM=1,1,", "  ISYM=1,", " &END"]
    assert lines[-1].split() == [f"{0.0:.16e}", "0", "0", "0", "0"]


def test_zero_integrals_are_omitted_and_ecore_is_written_last(tmp_path):
    zeros = {k: np.zeros_like(v) for k, v in uhf_tensors(2).items()}
    zeros["h1_b"][1, 0] = zeros["h1_b"][0, 1] = 0.5
    path = tmp_path / "fcidump.txt"
    mod.write_sbd_uhf_fcidump(path, norb=2, nelec=(1, 1), ecore=-3.25, **zeros)
    body = path.read_text().splitlines()[4:]
    assert [line.split()[1:] for line in body] == [["4", "2", "0", "0"], ["0", "0", "0", "0"]]
    assert float(body[-1].split()[0]) == -3.25


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.update(h1_a=np.eye(2)), "h1_a has shape"),
        (lambda t: t.update(h2_bb=t["h2_bb"] * 1j), "h2_bb is complex"),
        (lambda t: t["h1_a"].__setitem__((0, 0), np.nan), "h1_a contains non-finite"),
    ],
)
def test_check_rejects_blocks_the_file_cannot_represent(change, message):
    t = uhf_tensors(3)
    change(t)
    with pytest.raises(ValueError, match=message):
        mod.check_unrestricted_integrals(norb=3, **t)


def test_check_accepts_symmetric_blocks():
    mod.check_unrestricted_integrals(norb=3, **uhf_tensors(3))


def test_read_build_info(tmp_path):
    assert mod.read_build_info(tmp_path) is None
    (tmp_path / mod.BUILD_INFO_FILE).write_text("integrals=SBD_UHF_INTERLEAVED\n")
    assert mod.read_build_info(tmp_path) == mod.SBD_UHF_INTERLEAVED
