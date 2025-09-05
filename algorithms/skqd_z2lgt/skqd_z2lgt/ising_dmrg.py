"""Wrapper for ising_dmrg.jl."""
import os
import sys
import tempfile
import subprocess
from typing import Optional
import numpy as np
import h5py
from qiskit.quantum_info import SparsePauliOp


def ising_dmrg(
    hamiltonian: SparsePauliOp,
    filename: Optional[str] = None,
    julia_bin: str = 'julia'
):
    """Call ising_dmrg.jl. There must be a much smarter way to do this."""
    zz_indices = []
    zz_coeffs = []
    z_indices = []
    z_coeffs = []
    x_indices = []
    x_coeffs = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        pstr = pauli.to_label()
        if pstr.count('Z') == 2:
            first = pstr.index('Z')
            zz_indices.append([first, pstr.index('Z', first + 1)])
            zz_coeffs.append(coeff)
        elif pstr.count('Z') == 1:
            z_indices.append(pstr.index('Z'))
            z_coeffs.append(coeff)
        elif pstr.count('X') == 1:
            x_indices.append(pstr.index('X'))
            x_coeffs.append(coeff)
        else:
            raise RuntimeError(pstr)

    is_tempfile = False
    if not filename:
        is_tempfile = True
        with tempfile.NamedTemporaryFile() as tfile:
            filename = tfile.name

    with h5py.File(filename, 'w') as out:
        out.create_dataset('num_qubits', data=hamiltonian.num_qubits)
        out.create_dataset('zz_indices', data=np.array(zz_indices))
        out.create_dataset('zz_coeffs', data=np.array(zz_coeffs))
        out.create_dataset('z_indices', data=np.array(z_indices))
        out.create_dataset('z_coeffs', data=np.array(z_coeffs))
        out.create_dataset('x_indices', data=np.array(x_indices))
        out.create_dataset('x_coeffs', data=np.array(x_coeffs))

    program = os.path.join(
        os.path.dirname(__file__),
        '_julia',
        'ising_dmrg.jl'
    )

    proc = subprocess.run([julia_bin, program, filename],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    with h5py.File(filename, 'r') as source:
        energy = source['energy'][()]

    if is_tempfile:
        os.unlink(filename)

    return energy
