"""Unrestricted (UHF) integral input for an SBD driver built with ``-D_UHF``.

``-D_UHF`` is a compile-time switch of SBD: such a binary reads the FCIDUMP in an
interleaved spin-orbital layout, while the default build reads the usual restricted
(spatial-orbital) FCIDUMP. Neither layout is self-describing, and a ``-D_UHF`` binary given a
restricted FCIDUMP runs to completion with a wrong Hamiltonian. The writer here therefore
adds a header key, ``QCSC_INTEGRALS``, and ``native/main.cc`` refuses a file whose layout does
not match its build. A restricted FCIDUMP without the key is a restricted file, so FCIDUMPs
written before this check keep working with restricted binaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INTEGRALS_KEY = "QCSC_INTEGRALS"
"""FCIDUMP header key declaring the integral layout."""

RESTRICTED = "RESTRICTED"
"""Layout of the default SBD build (spatial orbitals, alpha and beta share the integrals)."""

SBD_UHF_INTERLEAVED = "SBD_UHF_INTERLEAVED"
"""Layout of the ``-D_UHF`` build: interleaved spin orbitals, alpha = 2p+1, beta = 2p+2."""

BUILD_INFO_FILE = "sbd_build_info.txt"
"""File in which ``native/main.cc`` reports its integral layout as ``integrals=<layout>``."""

LAYOUTS = {"restricted": RESTRICTED, "unrestricted": SBD_UHF_INTERLEAVED}
"""``SBDSolverJob.integrals`` value to FCIDUMP / binary layout."""


def write_sbd_uhf_fcidump(
    path: Path,
    *,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float = 0.0,
) -> None:
    """Write an interleaved-spin-orbital FCIDUMP for the SBD ``-D_UHF`` build.

    Ported unchanged from ``_write_uhf_fcidump`` of the UHF-SQD source
    (kunalroy332/qcsc-prefect@3968e585f7ca8137c127e5ee9f4fb383322d3d0d,
    ``algorithms/sbd/sbd/solver_job.py``), except that the first header line also declares
    ``QCSC_INTEGRALS=SBD_UHF_INTERLEAVED``. The integral records are identical.

    The ``_UHF`` ``SetupIntegrals`` parser does NOT use PySCF's four-block MOLPRO UHF format.
    It indexes integrals by spin-orbital with alpha = even 0-based index (``2*p``) and beta =
    odd (``2*p + 1``); the FCIDUMP records are 1-based over ``1..2*norb``. The spin block of a
    two-body record ``(i,j,k,l)`` is determined by ``S = (i-1)%2 + 2*((k-1)%2)`` and requires
    ``(i-1)%2 == (j-1)%2`` and ``(k-1)%2 == (l-1)%2`` (mixed-spin pair terms are zero):

        S=0  (alpha,alpha | alpha,alpha)  -> h2_aa
        S=1  (beta ,beta  | alpha,alpha)  -> h2_ab transposed (chem. (kl|ij) symmetry)
        S=2  (alpha,alpha | beta ,beta )  -> h2_ab
        S=3  (beta ,beta  | beta ,beta )  -> h2_bb

    Tensors are chemist-notation ``(ij|kl)`` as returned by PySCF ``ao2mo``. The reader applies
    its own 8-fold symmetrization, so we emit each unique (i<=j, k<=l, ij<=kl) record once.
    ``ecore`` is the constant term of the file (the solver adds it to its energy); the solver
    job writes 0 and the caller adds its own constant once.
    """
    nelec_a, nelec_b = nelec

    lines: list[str] = []
    lines.append(
        f" &FCI NORB={norb},NELEC={nelec_a + nelec_b},MS2={nelec_a - nelec_b},"
        f"{INTEGRALS_KEY}={SBD_UHF_INTERLEAVED},"
    )
    lines.append("  ORBSYM=" + ",".join(["1"] * norb) + ",")
    lines.append("  ISYM=1,")
    lines.append(" &END")

    def spinorb(spatial: int, spin_is_beta: bool) -> int:
        # 1-based spin-orbital index: alpha -> 2*p+1, beta -> 2*p+2 (p 0-based spatial).
        return 2 * spatial + (2 if spin_is_beta else 1)

    def emit_two_body(tensor: np.ndarray, bra_beta: bool, ket_beta: bool) -> None:
        # Chemist (ij|kl). Same-spin blocks (aa|aa, bb|bb) have full 8-fold symmetry, so emit only
        # i>=j, k>=l, (i,j)>=(k,l). The mixed block (aa|bb) only has the 4-fold permutation
        # symmetry i<->j and k<->l (the bra<->ket swap maps it to the distinct bb|aa block, which
        # the reader reconstructs by its own symmetrization), so for it we must NOT apply the
        # (i,j)>=(k,l) cut or we drop valid records like (00|11).
        same_spin = bra_beta == ket_beta
        for i in range(norb):
            for j in range(i + 1):
                ij = i * (i + 1) // 2 + j
                for k in range(norb):
                    for m in range(k + 1):
                        kl = k * (k + 1) // 2 + m
                        if same_spin and ij < kl:
                            continue
                        val = float(tensor[i, j, k, m])
                        if val == 0.0:
                            continue
                        si = spinorb(i, bra_beta)
                        sj = spinorb(j, bra_beta)
                        sk = spinorb(k, ket_beta)
                        sl = spinorb(m, ket_beta)
                        lines.append(f"{val:>28.16e} {si:4d} {sj:4d} {sk:4d} {sl:4d}")

    # Two-body blocks: S=0 aa|aa, S=2 aa|bb, S=3 bb|bb. The S=1 bb|aa block is the (kl|ij)
    # transpose of aa|bb and is recovered by the reader's symmetrization, so we emit aa|bb only.
    emit_two_body(h2_aa, bra_beta=False, ket_beta=False)
    emit_two_body(h2_ab, bra_beta=False, ket_beta=True)
    emit_two_body(h2_bb, bra_beta=True, ket_beta=True)

    # One-body: alpha block at even spin-orbitals, beta at odd. k=l=0 marks a one-body record.
    def emit_one_body(h1: np.ndarray, beta: bool) -> None:
        for i in range(norb):
            for j in range(i + 1):
                val = float(h1[i, j])
                if val == 0.0:
                    continue
                si = spinorb(i, beta)
                sj = spinorb(j, beta)
                lines.append(f"{val:>28.16e} {si:4d} {sj:4d} {0:4d} {0:4d}")

    emit_one_body(h1_a, beta=False)
    emit_one_body(h1_b, beta=True)

    # Core energy record (i=j=k=l=0).
    lines.append(f"{float(ecore):>28.16e} {0:4d} {0:4d} {0:4d} {0:4d}")

    Path(path).write_text("\n".join(lines) + "\n")


def check_unrestricted_integrals(
    *,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    atol: float = 1e-10,
) -> None:
    """Reject alpha/beta integral blocks the interleaved FCIDUMP cannot represent.

    The writer emits only the lower triangle of each permutation-symmetric block and SBD
    rebuilds the rest, so a block that is not symmetric would be changed silently. Each
    block must be real, finite and of shape ``(norb, norb)`` or ``(norb,) * 4``; ``h1_*`` and the
    same-spin ``h2_aa`` / ``h2_bb`` must have the 2- and 8-fold symmetry of real orbitals, and
    ``h2_ab`` the ``i<->j``, ``k<->l`` symmetry, within ``atol``.
    """
    blocks = {"h1_a": h1_a, "h1_b": h1_b, "h2_aa": h2_aa, "h2_ab": h2_ab, "h2_bb": h2_bb}
    for name, value in blocks.items():
        array = np.asarray(value)
        shape = (norb, norb) if name.startswith("h1") else (norb,) * 4
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {shape} for norb={norb}.")
        if np.iscomplexobj(array):
            raise ValueError(f"{name} is complex; the SBD -D_UHF build reads real integrals.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
    checks = [
        ("h1_a", np.asarray(h1_a), np.asarray(h1_a).T),
        ("h1_b", np.asarray(h1_b), np.asarray(h1_b).T),
    ]
    for name in ("h2_aa", "h2_bb"):
        t = np.asarray(blocks[name])
        checks += [
            (name, t, t.transpose(1, 0, 2, 3)),
            (name, t, t.transpose(0, 1, 3, 2)),
            (name, t, t.transpose(2, 3, 0, 1)),
        ]
    t = np.asarray(h2_ab)
    checks += [("h2_ab", t, t.transpose(1, 0, 2, 3)), ("h2_ab", t, t.transpose(0, 1, 3, 2))]
    for name, a, b in checks:
        if not np.allclose(a, b, rtol=0.0, atol=atol):
            raise ValueError(
                f"{name} lacks the permutation symmetry of real orbitals (max deviation "
                f"{float(np.max(np.abs(a - b))):.3e} > {atol}); the interleaved FCIDUMP would "
                "keep only its lower triangle."
            )


def read_build_info(work_dir: Path) -> str | None:
    """Integral layout reported by the driver in ``sbd_build_info.txt``, or None if absent."""
    path = Path(work_dir) / BUILD_INFO_FILE
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "integrals":
            return value.strip()
    return None
