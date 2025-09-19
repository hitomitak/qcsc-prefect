"""Definition of SKQD workflow."""

import asyncio

import ffsim
import numpy as np
import pyscf
import pyscf.mcscf
from ffsim.qiskit import PRE_INIT, jordan_wigner
from ffsim import MolecularHamiltonian, fermion_operator
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit.runtime import QuantumRuntime
from pydantic import BaseModel, Field
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import Target, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit.primitives import BitArray
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from prefect_dice import DiceSHCISolverJob

from .subsample import postselect, subsample


class Parameters(BaseModel):
    """Workflow parameters to configure skqd_2501_09702 workflow."""

    atom: str = Field(
        description="Definition for molecule structure.",
        title="Atom",
    )
    spin_sq: float = Field(
        default=0.0,
        description="Target value for the total spin squared for the ground state.",
        title="Total Spin Squared",
    )
    basis: str = Field(
        default="6-31g",
        description="Name of basis set.",
        title="Basis Set",
    )
    n_frozen: int = Field(
        default=0,
        description="Number of non-excitation orbitals.",
        title="Frozen Orbitals",
        ge=0,
    )
    symmetry: str | bool = Field(
        default=False,
        description="Whether to use symmetry, otherwise string of point group name.",
        title="Symmetry",
    )
    symmetrize_spin: bool = Field(
        default=True,
        description="Whether to always merge spin-alpha and spin-beta CI strings into a single list.",
        title="Symmetrize Spin",
    )
    skqd_subspace_dim: int = Field(
        description="SKQD: Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )
    skqd_num_batches: int = Field(
        description="SKQD: Number of batches of configurations used by the different calls to the eigenstate solver.",
        title="Batch Number",
        ge=1,
    )
    skqd_max_iterations: int = Field(
        description="SKQD: Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
    )
    krylov_dim: int = Field(
        default=5,
        desciption="Size of Krylov subspace",
        title="Krylov Dimension",
        ge=1,
    )
    dt: float = Field(
        default=0.15,
        description="SKQD: Time Interval for Trotterization.",
        title="Time Interval",
        ge=0,
    )
    n_trotter_steps: int = Field(
        default=6,
        description="SKQD: Number of Trotter steps",
        title="Trotter Steps",
        ge=1,
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )
    sabre_max_iterations: int = Field(
        default=8,
        description="Transpile: The number of forward-backward routing iterations to refine the layout and reduce routing costs.",
        title="SABRE Max Iteration",
        ge=1,
    )
    sabre_swap_trials: int = Field(
        default=10,
        description="Transpile: The number of routing trials for each layout, refining gate placement for better routing.",
        title="SABRE SWAP Trials",
        ge=1,
    )
    sabre_layout_trials: int = Field(
        default=10,
        description="Transpile: The number of random initial layouts tested, selecting the one that minimizes SWAP gates.",
        title="SABRE Layout Trials",
        ge=1,
    )


@flow
async def skqd_2501_09702(
    parameters: Parameters,
    runner_name: str = "skqd-runner",
    solver_name: str = "skqd-solver",
):
    """SKQD Experiment from arXiv:2501.09702."""
    rng = np.random.default_rng(24)
    logger = get_run_logger()

    # Compute molecular integrals
    hcore, eri, t2, nuclear_repulsion_energy, norb, nelec = compute_molecular_integrals(
        parameters
    )

    # Sample bitstrings
    try:
        # Run on a real hardware when credentials are found.
        runtime = await QuantumRuntime.load(runner_name)
        options = await Variable.get("sampler_options")

        # ansatz_circuit = create_ansatz_circuits(
        #     parameters=parameters,
        #     one_body_tensor=hcore,
        #     num_electrons=nelec,
        #     t2=t2,
        # )
        trotter_circuits = create_trotter_circuits(
            parameters=parameters,
            one_body_tensor=hcore,
            two_body_tensor=eri,
            num_electrons=nelec,
        )
        isa_circuits = transpile_circuits(
            circuits=trotter_circuits,
            target=await runtime.get_target(),
            parameters=parameters,
            seed=rng,
        )
        pub_results = await runtime.sampler(
            sampler_pubs=[(isa_circuit,) for isa_circuit in isa_circuits],
            options=options,
        )
        # bit_array = pub_results[0].data.meas

        # Combine the counts from the individual Trotter circuits
        bit_array = BitArray.concatenate_shots(
            [result.data.meas for result in pub_results]
        )

    except ValueError:
        # Uniform sampling when runtime is not defined.
        logger.warning(
            f"Runner block {runner_name} is not defined. "
            "Falling back into random uniform sampling."
        )
        bit_array = generate_bit_array_uniform(10_000, norb * 2, rand_seed=rng)

    # Convert a counts dictionary into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    n_alpha, n_beta = nelec

    # Run configuration recovery loop
    sci_solver = await DiceSHCISolverJob.load(solver_name)
    skqd_artifact = []
    current_occupancies = None
    best_result = None
    for round_idx in range(parameters.skqd_max_iterations):
        if current_occupancies is None:
            bitstrings, probs = raw_bitstrings, raw_probs
        else:
            bitstrings, probs = recover_configurations(
                raw_bitstrings,
                raw_probs,
                current_occupancies,
                n_alpha,
                n_beta,
                rand_seed=rng,
            )
        bitstrings, probs = postselect(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            hamming_right=n_alpha,
            hamming_left=n_beta,
        )
        # Convert bitstrings to CI strings and diagonalize
        coros = []
        subspace_dims = []
        for ci_strings in subsample(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            subspace_dim=parameters.skqd_subspace_dim,
            num_batches=parameters.skqd_num_batches,
            rng=rng,
            open_shell=not parameters.symmetrize_spin,
        ):
            subspace_dims.append(int(len(ci_strings[0]) * len(ci_strings[1])))
            coro = sci_solver.run(
                ci_strings=ci_strings,
                one_body_tensor=hcore,
                two_body_tensor=eri,
                norb=norb,
                nelec=nelec,
                spin_sq=parameters.spin_sq,
            )
            coros.append(coro)
        results: list[SCIResult] = await asyncio.gather(*coros)
        # Get best result from batch
        best_result_in_batch = min(results, key=lambda result: result.energy)
        # Check if the energy is the lowest seen so far
        if best_result is None or best_result_in_batch.energy < best_result.energy:
            best_result = best_result_in_batch
        current_occupancies = best_result_in_batch.orbital_occupancies
        # Update artifact
        for subspace_idx, result in enumerate(results):
            new_record = dict(
                round=int(round_idx),
                subspace=int(subspace_idx),
                energy=float(result.energy + nuclear_repulsion_energy),
                dimension=subspace_dims[subspace_idx],
            )
            skqd_artifact.append(new_record)
        logger.info(
            f"Best energy in SKQD iteration {round_idx}: {best_result_in_batch.energy + nuclear_repulsion_energy} Ha."
        )

    # Upload artifact
    await create_table_artifact(
        table=skqd_artifact,
        key="skqd-configuration-recovery",
    )
    return best_result.energy + nuclear_repulsion_energy


@task
def compute_molecular_integrals(
    parametrers: Parameters,
) -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int]]:
    """
    Compute molecular property using the classical method.

    This function computes molecular integrals and prepares the necessary components for further quantum chemical calculations.

    Parameters:
    parametrers (Parameters): An object containing molecule specifications such as atom, basis, symmetry, n_frozen, etc.

    Returns:
    tuple: A tuple containing:
        - hcore (np.ndarray): One-electron core Hamiltonian matrix.
            - Shape: (n_orb, n_orb) where n_orb is the number of active orbitals.
        - eri (np.ndarray): Two-electron integral matrix.
            - Shape: (n_orb, n_orb, n_orb, n_orb)
        - ccsd.t2 (float): Amplitudes for the CCSD t2 equation.
        - nuclear_repulsion_energy (int): Nuclear repulsion energy of the molecule.
        - num_orbitals (int): Number of active orbitals.
        - electron_count (tuple[int, int]): Number of alpha and beta electrons.
    """
    # Specify molecule properties
    mol = pyscf.gto.Mole()
    mol.build(
        atom=parametrers.atom,
        basis=parametrers.basis,
        symmetry=parametrers.symmetry,
    )
    # Define active space
    active_space = range(parametrers.n_frozen, mol.nao_nr())
    # Get molecular integrals
    scf = pyscf.scf.RHF(mol).run()
    num_orbitals = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    num_elec_a = (n_electrons + mol.spin) // 2
    num_elec_b = (n_electrons - mol.spin) // 2
    cas = pyscf.mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)
    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf,
        frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
    ).run()

    return (
        hcore,
        eri,
        ccsd.t2,
        nuclear_repulsion_energy,
        num_orbitals,
        (num_elec_a, num_elec_b),
    )


@task
def create_trotter_circuits(
    parameters: Parameters,
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    num_electrons: tuple[int, int],
) -> QuantumCircuit:
    """Create Trotterized circuit.
    See: https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/skqd#1-map-problem-to-quantum-circuits-and-operators
    """
    num_orbitals = one_body_tensor.shape[0]
    logger = get_run_logger()

    qreg = QuantumRegister(2 * num_orbitals, name="q")
    qc_state_prep = QuantumCircuit(qreg)
    # Prepare HF state as initial state
    qc_state_prep.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=num_orbitals,
            nelec=num_electrons,
        ),
        qargs=qreg,
    )

    # Get MolecularHamiltonian
    mol_hamiltonian = MolecularHamiltonian(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
    )
    # Convert MolecularHamiltonian to FermionOperator
    ferm_op = fermion_operator(mol_hamiltonian)

    # Convert FermionOperator to SparsePauliOp
    H_op = jordan_wigner(ferm_op, norb=num_orbitals)

    # `U` operator
    evol_gate = PauliEvolutionGate(
        H_op,
        time=(parameters.dt / parameters.n_trotter_steps),
        synthesis=LieTrotter(reps=parameters.n_trotter_steps),
    )
    qc_evol = QuantumCircuit(qreg)
    qc_evol.append(evol_gate, qargs=qreg)

    circuits = []
    for rep in range(parameters.krylov_dim):
        circ = qc_state_prep.copy()

        # Repeating the `U` operator to implement U^0, U^1, U^2, and so on, for power Krylov space
        for _ in range(rep):
            circ.compose(other=qc_evol, inplace=True)

        circ.measure_all()
        logger.info(
            f"{rep} th untranspiled circuit:\n"
            f"Number of qubits = {circ.num_qubits}\n"
            f"Number of classical bits = {circ.num_clbits}\n"
            f"Circuit depth = {circ.depth()}\n"
            f"Instruction counts = {dict(circ.count_ops())}\n"
            f"--------------------------------\n"
        )
        circuits.append(circ)

    return circuits


@task
def transpile_circuits(
    circuits: list[QuantumCircuit],
    target: Target,
    parameters: Parameters,
    seed: np.random.Generator,
) -> QuantumCircuit:
    """Custom transpiler with massive layout search."""
    logger = get_run_logger()

    if isinstance(seed, np.random.Generator):
        seed = seed.integers(np.iinfo(int).max, size=1).item()
    coupling_map = target.build_coupling_map()

    # Define a custom pass manager
    passmanager = generate_preset_pass_manager(
        optimization_level=parameters.optimization_level,
        seed_transpiler=seed,
        target=target,
    )
    passmanager.pre_init = PRE_INIT
    passmanager.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=seed,
                max_iterations=parameters.sabre_max_iterations,
                layout_trials=parameters.sabre_layout_trials,
                swap_trials=parameters.sabre_swap_trials,
            ),
            ConditionalController(
                tasks=[
                    FullAncillaAllocation(coupling_map=coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda propset: propset["final_layout"] is None,
            ),
        ]
    )
    if "rzz" in target.operation_names:
        passmanager.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )

    # Transpile
    isa_circuits = []
    for rep, circuit in enumerate(circuits):
        isa_circuit = passmanager.run(circuit)
        gate_depth = isa_circuit.depth(
            lambda inst: inst.operation.name not in ("rz", "barrier", "measure")
        )
        logger.info(
            f"{rep} th transpiled circuit:\n"
            f"Number of qubits = {isa_circuit.num_qubits}\n"
            f"Number of classical bits = {isa_circuit.num_clbits}\n"
            f"Circuit depth = {gate_depth}\n"
            f"Instruction counts = {dict(isa_circuit.count_ops())}\n"
            f"--------------------------------\n"
        )
        isa_circuits.append(isa_circuit)
    return isa_circuits


def deploy():
    """Deploy workflow with a local worker."""
    import os
    import pathlib

    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    # Deploy the workflow with specified options.
    skqd_2501_09702.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="skqd_2501_09702",
        description="SKQD experiment from arXiv:2501.09702.",
    )
