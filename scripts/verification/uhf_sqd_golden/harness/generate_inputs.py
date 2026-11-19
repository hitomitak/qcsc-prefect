"""Generate the PR0 input FCIDUMPs (restricted MO basis, all orbitals active, ECORE = E_nuc).

Usage: python -I generate_inputs.py OUTDIR
NH2 / STO-3G: ROHF orbitals, doublet (5a, 4b).
H6 / STO-3G: linear chain, 2.0 A spacing, RHF orbitals, singlet (3a, 3b); the branch UHF path is
expected to find a broken-symmetry solution from it.
"""

import hashlib
import json
import sys
from pathlib import Path

from pyscf import ao2mo, gto, scf
from pyscf.tools import fcidump

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
systems = {
    "nh2_sto3g": dict(
        atom="N 0 0 0; H 0 0.80 0.59; H 0 -0.80 0.59", basis="sto-3g", spin=1, scf="ROHF"
    ),
    "h6_chain2A_sto3g": dict(
        atom="; ".join(f"H 0 0 {2.0 * i}" for i in range(6)), basis="sto-3g", spin=0, scf="RHF"
    ),
}
meta = {}
for name, s in systems.items():
    mol = gto.M(atom=s["atom"], basis=s["basis"], spin=s["spin"], unit="Angstrom", verbose=0)
    mf = getattr(scf, s["scf"])(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    h2 = ao2mo.full(mol, c)
    path = out / f"{name}.fcidump"
    fcidump.from_integrals(str(path), h1, h2, mol.nao, mol.nelec, nuc=mol.energy_nuc())
    meta[name] = {
        **s,
        "nao": mol.nao,
        "nelec": list(mol.nelec),
        "e_nuc": mol.energy_nuc(),
        "e_scf": mf.e_tot,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
(out / "inputs.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
