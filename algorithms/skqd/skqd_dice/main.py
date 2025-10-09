"""Definition of SKQD workflow."""

import asyncio
import os
import pathlib
import ffsim
import numpy as np
from ffsim.qiskit import PRE_INIT, jordan_wigner
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import NO_CACHE, INPUTS
from prefect.variables import Variable
from prefect_qiskit.runtime import QuantumRuntime
from pydantic import BaseModel, Field
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import Target
from ffsim import MolecularHamiltonian, fermion_operator
from qiskit.passmanager import ConditionalController
from qiskit.primitives.containers import BitArray
from qiskit.transpiler import generate_preset_pass_manager
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
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from prefect_dice import DiceSHCISolverJob

from skqd_dice.subsample import postselect, subsample

from qcsc_workflow_utility import (
    ElectronicProperties,
    compute_molecular_integrals_from_geometry,
    compute_molecular_integrals_from_fcidump,
)


class MoleculeGeometry(BaseModel):
    """Molecule definition by geometory and basis."""

    atom: str = Field(
        description="Definition for molecule structure.",
        title="Atom",
    )
    basis: str = Field(
        default="6-31g",
        description="Name of basis set.",
        title="Basis Set",
    )
    symmetry: str | bool = Field(
        default=False,
        description="Whether to use symmetry, otherwise string of point group name.",
        title="Symmetry",
    )
    spin_sq: float = Field(
        default=0.0,
        description="Target value for the total spin squared for the ground state.",
        title="Total Spin Squared",
    )


class FCIDumpFile(BaseModel):
    """Molecule definition by FCIDump file."""

    fcidump_file: str = Field(
        default="",
        description=(
            "Location of FCIDump file storing 1-electron and 2-electron integrals. "
            "We assume these are precomputed in the MO basis."
        ),
        title="FCIDump File",
    )
    spin_sq: float = Field(
        default=0.0,
        description="Target value for the total spin squared for the ground state.",
        title="Total Spin Squared",
    )


class CircuitParameters(BaseModel):
    """Configuration for circuit."""

    use_reset_mitigation: bool = Field(
        default=False,
        description=(
            "Use reset mitigation scheme that post-selects outcomes with non-ground initial state. "
            "This post-selection reduces the net shot number and its retention rate depends on "
            "the quality of hardware reset instruction."
        ),
        title="Reset Mitigation",
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
        default=10000,
        description="Transpile: The number of random initial layouts tested, selecting the one that minimizes SWAP gates.",
        title="SABRE Layout Trials",
        ge=1,
    )


class SKQDParameters(BaseModel):
    """Configuration for SKQD algorithm execution."""

    n_trotter_steps: int = Field(
        default=2,
        description="Number of Trotter steps.",
        title="Trotter Steps",
        ge=1,
    )
    dt: float = Field(
        default=0.15,
        description="Time Interval for Trotterization.",
        title="Time Interval",
        ge=0,
    )
    krylov_dim: int = Field(
        default=5,
        desciption="Size of Krylov subspace",
        title="Krylov Dimension",
        ge=1,
    )
    subspace_dim: int = Field(
        description="Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )
    num_batches: int = Field(
        description="Number of batches of configurations used by the different calls to the eigenstate solver.",
        title="Batch Number",
        ge=1,
    )
    max_iterations: int = Field(
        description="Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
    )


class Parameters(BaseModel):
    """Workflow parameters to configure skqd_2501_09702 workflow."""

    molecule: MoleculeGeometry | FCIDumpFile = Field(
        default_factory=MoleculeGeometry,
        description="PySCF definition of the molecule to solve for.",
        title="Molecule",
    )

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description="Ansatz circuit definition and transpiler settings.",
        title="Circuit",
    )

    skqd: SKQDParameters = Field(
        default_factory=SKQDParameters,
        description="Control of SKQD algorithm execution and solver parameters.",
        title="SKQD",
    )


@flow
async def skqd_2501_09702(
    parameters: Parameters,
    runner_name: str = "skqd-runner",
    solver_name: str = "skqd-solver",
    option_name: str = "sampler_options",
    cache_compute_integrals: bool = False,
    cache_sampling: bool = False,
):
    """SKQD Experiment from arXiv:2501.09702.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRunner block to load.
        solver_name: Name of DiceSHCISolverJob block to load.
        option_name: Name of Variable storing sampler primitive options to load.
        cache_compute_integrals: Set True to cache compute_molecular_integrals task.
        cache_sampling: Set True to cache sample_bitstrings task.

    Returns:
        Minimum ground state energy solved by SKQD.
    """
    rng = np.random.default_rng(24)
    logger = get_run_logger()

    # Compute molecular integrals
    if isinstance(parameters.molecule, MoleculeGeometry):
        elec_props = compute_molecular_integrals_from_geometry.with_options(
            cache_policy=INPUTS if cache_compute_integrals else NO_CACHE,
        )(
            atom=parameters.molecule.atom,
            basis=parameters.molecule.basis,
            symmetry=parameters.molecule.symmetry,
            spin_sq=parameters.molecule.spin_sq,
        )
    elif isinstance(parameters.molecule, FCIDumpFile):
        elec_props = compute_molecular_integrals_from_fcidump.with_options(
            cache_policy=INPUTS if cache_compute_integrals else NO_CACHE,
        )(
            fcidump_file=parameters.molecule.fcidump_file,
            spin_sq=parameters.molecule.spin_sq,
        )
    else:
        raise TypeError("Unsupported molecule type")

    # sample bitstrings
    bit_array = await sample_bitstrings.with_options(
        cache_policy=INPUTS if cache_sampling else NO_CACHE,
    )(
        parameters=parameters,
        elec_props=elec_props,
        runner_name=runner_name,
        option_name=option_name,
    )

    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    n_alpha, n_beta = elec_props.num_electrons

    # Run configuration recovery loop
    sci_solver = await DiceSHCISolverJob.load(solver_name)
    skqd_artifact = []
    current_occupancies = elec_props.initial_occupancy
    best_result = None
    for round_idx in range(parameters.skqd.max_iterations):
        bitstrings, probs = recover_configurations(
            raw_bitstrings,
            raw_probs,
            current_occupancies,
            n_alpha,
            n_beta,
            rand_seed=rng,
        )
        bitstrings_post, probs_post = postselect(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            hamming_right=n_alpha,
            hamming_left=n_beta,
        )
        logger.info(f"Number of post-selected bitstrings: {len(bitstrings_post)}")
        if len(bitstrings_post) == 0:
            raise RuntimeError(
                "The input bit array did not contain any valid bitstrings. "
                "Pass a bit array that contains at least one valid bitstring."
            )
        # Convert bitstrings to CI strings and diagonalize
        coros = []
        subspace_dims = []
        for ci_strings in subsample(
            bitstring_matrix=bitstrings_post,
            probabilities=probs_post,
            subspace_dim=parameters.skqd.subspace_dim,
            num_batches=parameters.skqd.num_batches,
            rng=rng,
            open_shell=elec_props.open_shell,
        ):
            subspace_dims.append(int(len(ci_strings[0]) * len(ci_strings[1])))
            coro = sci_solver.run(
                ci_strings=ci_strings,
                one_body_tensor=elec_props.one_body_tensor,
                two_body_tensor=elec_props.two_body_tensor,
                norb=elec_props.num_orbitals,
                nelec=elec_props.num_electrons,
                spin_sq=elec_props.spin_sq,
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
                energy=float(result.energy + elec_props.nuclear_repulsion_energy),
                dimension=subspace_dims[subspace_idx],
            )
            skqd_artifact.append(new_record)
        logger.info(
            f"Best energy in SKQD iteration {round_idx}: {best_result_in_batch.energy + elec_props.nuclear_repulsion_energy} Ha."
        )

    # Upload artifact
    await create_table_artifact(
        table=skqd_artifact,
        key="skqd-configuration-recovery",
    )
    return best_result.energy + elec_props.nuclear_repulsion_energy


@task
def create_trotter_circuits(
    skqd_params: SKQDParameters,
    elec_props: ElectronicProperties,
) -> list[QuantumCircuit]:
    """Create Trotterized circuit.
    See: https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/skqd#1-map-problem-to-quantum-circuits-and-operators
    """
    num_orbitals = elec_props.num_orbitals
    one_body_tensor = elec_props.one_body_tensor
    two_body_tensor = elec_props.two_body_tensor
    num_electrons = elec_props.num_electrons
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
        time=(skqd_params.dt / skqd_params.n_trotter_steps),
        synthesis=LieTrotter(reps=skqd_params.n_trotter_steps),
    )
    qc_evol = QuantumCircuit(qreg)
    qc_evol.append(evol_gate, qargs=qreg)

    circuits = []
    for rep in range(skqd_params.krylov_dim):
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
    circuit_params: CircuitParameters,
    seed: np.random.Generator,
) -> list[QuantumCircuit]:
    """Custom transpiler with massive layout search."""
    logger = get_run_logger()

    if isinstance(seed, np.random.Generator):
        seed = seed.integers(np.iinfo(int).max, size=1).item()
    coupling_map = target.build_coupling_map()

    # Define a custom pass manager
    passmanager = generate_preset_pass_manager(
        optimization_level=circuit_params.optimization_level,
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
                max_iterations=circuit_params.sabre_max_iterations,
                layout_trials=circuit_params.sabre_layout_trials,
                swap_trials=circuit_params.sabre_swap_trials,
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
    logger.info("Transpiling ansatz circuit...")
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


@task(
    persist_result=True,
    result_serializer="compressed/pickle",
)
async def sample_bitstrings(
    parameters: Parameters,
    elec_props: ElectronicProperties,
    runner_name: str,
    option_name: str,
) -> BitArray:
    """Sample bitstring with quantum computer.

    Args:
        parameters: Workflow parameters including circuit and SKQD configuration.
        elec_props: Electronic properties of molecule to solver for.
        runner_name: Block name of QuantumRuntime to execute primitives on.
        option_name: Name of Variable storing sampler primitive options to load.

    Returns:
        Sampled bitstrings representing determinants.
    """
    logger = get_run_logger()
    rng = np.random.default_rng(24)

    try:
        # Run on a real hardware when credentials are found.
        runtime = await QuantumRuntime.load(runner_name)
        options = await Variable.get(option_name)

        # Create Trotter circuits
        trotter_circuits = create_trotter_circuits(
            skqd_params=parameters.skqd,
            elec_props=elec_props,
        )

        # Transpile
        target = await runtime.get_target()
        isa_circuits = transpile_circuits(
            circuits=trotter_circuits,
            target=target,
            circuit_params=parameters.circuit,
            seed=rng,
        )

        # Run primitive
        pub_results = await runtime.sampler(
            sampler_pubs=[(isa_circuit,) for isa_circuit in isa_circuits],
            options=options,
        )

        # Post-process bitstrings
        bit_arrays = []
        for result in pub_results:
            meas_bits = result.data.meas
            if parameters.circuit.use_reset_mitigation:
                test_bits = result.data.test
                bit_array = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
                bit_array = BitArray.from_samples(
                    bit_array, num_bits=meas_bits.num_bits
                )
                logger.info(
                    "Reset mitigation result:\n"
                    f"  Before: {meas_bits.num_shots} bitstrings\n"
                    f"  After: {bit_array.num_shots} bitstrings\n"
                    f"  Retention rate: {bit_array.num_shots / meas_bits.num_shots}\n"
                )
            else:
                bit_array = meas_bits
            bit_arrays.append(bit_array)

        # Combine the counts from the individual Trotter circuits
        bit_array = BitArray.concatenate_shots(bit_arrays)

    except ValueError:
        # Uniform sampling when runtime is not defined.
        logger.warning(
            f"QuantumRuntime block '{runner_name}' is not defined. "
            "Falling back into random uniform sampling."
        )
        return generate_bit_array_uniform(
            100_000,
            elec_props.num_orbitals * 2,
        )

    return bit_array


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    # Deploy the workflow with specified options.
    skqd_2501_09702.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="skqd_2501_09702",
        description="SKQD experiment from arXiv:2501.09702.",
    )
