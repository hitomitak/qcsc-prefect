"""DICE input/output helpers shared across algorithms."""

from __future__ import annotations

import logging
import math
import struct
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from uuid import uuid4

import jinja2
import numpy as np
from prefect import get_run_logger
from pyscf.tools import fcidump
from qiskit_addon_sqd.fermion import SCIResult, SCIState

MAX_DICE_DIMENSION = 2_147_483_647
DETERMINANT_BYTES = 16


def _logger():
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


def make_job_work_dir(base_work_dir: Path) -> Path:
    """Create a unique work directory for one DICE execution."""

    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _render_input_dat(
    *,
    spin_sq: float | None,
    select_cutoff: float,
    davidson_tol: float,
    energy_tol: float,
    max_iter: int,
    dim: int,
    num_elec: int,
) -> str:
    template_text = (
        resources.files("qcsc_prefect_dice")
        .joinpath("templates", "input.dat.j2")
        .read_text(encoding="utf-8")
    )
    env = jinja2.Environment()
    template = env.from_string(template_text)
    return template.render(
        spin_sq=spin_sq,
        select_cutoff=select_cutoff,
        davidson_tol=davidson_tol,
        energy_tol=energy_tol,
        max_iter=max_iter,
        dim=dim,
        num_elec=num_elec,
    )


def prep_dice_input_files(
    *,
    work_dir: Path,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    spin_sq: float | None,
    select_cutoff: float,
    davidson_tol: float,
    energy_tol: float,
    max_iter: int,
) -> None:
    """Prepare DICE input files in ``work_dir``."""

    logger = _logger()
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Writing fcidump.txt file.")
    fcidump.from_integrals(
        str(work_dir / "fcidump.txt"),
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
    )

    logger.debug("Writing input.dat file.")
    input_dat = _render_input_dat(
        spin_sq=spin_sq,
        select_cutoff=select_cutoff,
        davidson_tol=davidson_tol,
        energy_tol=energy_tol,
        max_iter=max_iter,
        dim=min(
            MAX_DICE_DIMENSION,
            math.comb(norb, nelec[0]) * math.comb(norb, nelec[1]),
        ),
        num_elec=nelec[0] + nelec[1],
    )
    (work_dir / "input.dat").write_text(input_dat, encoding="utf-8")

    logger.debug("Writing AlphaDets.bin and BetaDets.bin file.")
    with (work_dir / "AlphaDets.bin").open("wb") as fp:
        for ci in np.asarray(ci_strings[0]).reshape(-1):
            fp.write(int(ci).to_bytes(DETERMINANT_BYTES, byteorder="big", signed=False))
    with (work_dir / "BetaDets.bin").open("wb") as fp:
        for ci in np.asarray(ci_strings[1]).reshape(-1):
            fp.write(int(ci).to_bytes(DETERMINANT_BYTES, byteorder="big", signed=False))


def read_dice_output_files(
    *,
    work_dir: Path,
    norb: int,
    nelec: tuple[int, int],
    return_sci_state: bool,
) -> SCIResult:
    """Read DICE output files and reconstruct an ``SCIResult``."""

    logger = _logger()

    logger.debug("Reading spin1RDM file.")
    spin1_rdm_dice = np.loadtxt(work_dir / "spin1RDM.0.0.txt", skiprows=1)
    avg_occupancies = np.zeros(2 * norb)
    for i in range(spin1_rdm_dice.shape[0]):
        if spin1_rdm_dice[i, 0] == spin1_rdm_dice[i, 1]:
            orbital_id = int(spin1_rdm_dice[i, 0])
            parity = orbital_id % 2
            avg_occupancies[int(orbital_id // 2 + parity * norb)] = spin1_rdm_dice[i, 2]

    logger.debug("Reading shci.e file.")
    with (work_dir / "shci.e").open("rb") as fp:
        energy = struct.unpack("d", fp.read(8))[0]

    if not return_sci_state:
        logger.info("Skipping construction of SCIState object.")
        return SCIResult(
            energy=energy,
            sci_state=None,
            orbital_occupancies=(avg_occupancies[:norb], avg_occupancies[norb:]),
        )

    logger.debug("Reading dets.bin file.")
    occupancy_strs: list[str] = []
    amplitudes: list[float] = []
    with (work_dir / "dets.bin").open("rb") as fp:
        num_dets = struct.unpack("i", fp.read(4))[0]
        num_orb = struct.unpack("i", fp.read(4))[0]
        for i in range(2 * num_dets):
            if i % 2 == 0:
                amplitudes.append(struct.unpack("d", fp.read(8))[0])
            else:
                occupancy_strs.append(fp.read(num_orb).decode("ascii"))
    logger.info("Number of determinants in dets.bin file: %s", num_dets)

    ci_strs: list[tuple[int, int]] = []
    for occ_str in occupancy_strs:
        bitstring = np.zeros(2 * norb, dtype=bool)
        for i, bit in enumerate(occ_str):
            if bit == "2":
                bitstring[i] = True
                bitstring[i + norb] = True
            elif bit == "a":
                bitstring[i] = True
            elif bit == "b":
                bitstring[i + norb] = True
        ci_str_a = sum(int(b) << i for i, b in enumerate(bitstring[:norb]))
        ci_str_b = sum(int(b) << i for i, b in enumerate(bitstring[norb:]))
        ci_strs.append((ci_str_a, ci_str_b))

    if ci_strs:
        strs_a, strs_b = zip(*ci_strs)
        uniques_a = np.unique(strs_a)
        uniques_b = np.unique(strs_b)
        sci_coefficients = np.zeros((len(uniques_a), len(uniques_b)))
        ci_strs_a = np.zeros(len(uniques_a), dtype=np.int64)
        ci_strs_b = np.zeros(len(uniques_b), dtype=np.int64)
        ci_str_map_a = {uni_str: i for i, uni_str in enumerate(uniques_a)}
        ci_str_map_b = {uni_str: i for i, uni_str in enumerate(uniques_b)}
        for amp, ci_str in zip(amplitudes, ci_strs, strict=True):
            ci_str_a, ci_str_b = ci_str
            i = ci_str_map_a[ci_str_a]
            j = ci_str_map_b[ci_str_b]
            sci_coefficients[i, j] = amp
            ci_strs_a[i] = uniques_a[i]
            ci_strs_b[j] = uniques_b[j]
    else:
        sci_coefficients = np.zeros((0, 0))
        ci_strs_a = np.zeros(0, dtype=np.int64)
        ci_strs_b = np.zeros(0, dtype=np.int64)

    sci_state = SCIState(
        amplitudes=sci_coefficients,
        ci_strs_a=ci_strs_a,
        ci_strs_b=ci_strs_b,
        norb=norb,
        nelec=nelec,
    )
    return SCIResult(
        energy=energy,
        sci_state=sci_state,
        orbital_occupancies=(avg_occupancies[:norb], avg_occupancies[norb:]),
    )
