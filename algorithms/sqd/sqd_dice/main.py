"""Definition of SQD workflow."""

import asyncio
import io
from typing import Annotated

import ffsim
import numpy as np
import scipy
from pyscf import tools, ao2mo, cc, gto, scf
from ffsim.qiskit import PRE_INIT
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import NO_CACHE, INPUTS
from prefect.variables import Variable
from prefect_qiskit.runtime import QuantumRuntime
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
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

from sqd_dice.subsample import postselect, subsample


# Pydantic Types
NpStrict1DArrayF64 = Annotated[
    np.ndarray[tuple[int, ], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(
        data_type=np.float64, dimensions=1, strict_data_typing=True
    ),
]

NpStrict2DArrayF64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(
        data_type=np.float64, dimensions=2, strict_data_typing=True
    ),
]

NpStrict4DArrayF64 = Annotated[
    np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(
        data_type=np.float64, dimensions=4, strict_data_typing=True
    ),
]


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
    """Configuration for ansatz circuit."""

    n_lucj_layers: int = Field(
        default=1,
        description="Number of repetition of the unit layer of LUCJ ansatz.",
        title="LUCJ Layers",
        ge=1,
    )
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


class SQDParameters(BaseModel):
    """Configuration for SQD algorithm execution."""

    subspace_dim: int = Field(
        description="SQD: Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )
    num_batches: int = Field(
        description="SQD: Number of batches of configurations used by the different calls to the eigenstate solver.",
        title="Batch Number",
        ge=1,
    )
    max_iterations: int = Field(
        description="SQD: Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
    )


class Parameters(BaseModel):
    """Workflow parameters to configure sqd_2405_05068 workflow."""

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

    sqd: SQDParameters = Field(
        default_factory=SQDParameters,
        description="Control of SQD algorithm execution and solver parameters.",
        title="SQD",
    )


@dataclass
class ElectronicProperties:
    """Intermediate data representing the electronic properties."""

    one_body_tensor: NpStrict2DArrayF64
    two_body_tensor: NpStrict4DArrayF64
    t2: NpStrict4DArrayF64
    initial_occupancy: tuple[NpStrict1DArrayF64, NpStrict1DArrayF64]
    nuclear_repulsion_energy: float
    num_orbitals: int
    num_electrons: tuple[int, int]
    open_shell: bool
    spin_sq: float


@flow
async def sqd_2405_05068(
    parameters: Parameters,
    runner_name: str = "sqd-runner",
    solver_name: str = "sqd-solver",
    option_name: str = "sampler_options",
    cache_compute_integrals: bool = False,
    cache_sampling: bool = False,
) -> float:
    """SQD Experiment from arXiv2405.05068.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRunner block to load.
        solver_name: Name of DiceSHCISolverJob block to load.
        option_name: Name of Variable storing sampler primitive options to load.
        cache_compute_integrals: Set True to cache compute_molecular_integrals task.
        cache_sampling: Set True to cache sample_bitstrings task.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    rng = np.random.default_rng(24)
    logger = get_run_logger()

    # Compute integrals
    elec_props = compute_molecular_integrals.with_options(
        cache_policy=INPUTS if cache_compute_integrals else NO_CACHE,
    )(
        mol_params=parameters.molecule,
    )

    # Sample bitstrings
    bit_array = await sample_bitstrings.with_options(
        cache_policy=INPUTS if cache_sampling else NO_CACHE,
    )(
        circuit_params=parameters.circuit,
        elec_props=elec_props,
        runner_name=runner_name,
        option_name=option_name,
    )

    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    n_alpha, n_beta = elec_props.num_electrons

    # Run configuration recovery loop
    sci_solver = await DiceSHCISolverJob.load(solver_name)
    sqd_artifact = []
    current_occupancies = elec_props.initial_occupancy
    best_result = None
    for round_idx in range(parameters.sqd.max_iterations):
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
        logger.info(
            f"Number of post-selected bitstrings: {len(bitstrings_post)}"
        )
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
            subspace_dim=parameters.sqd.subspace_dim,
            num_batches=parameters.sqd.num_batches,
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
            sqd_artifact.append(new_record)
        logger.info(
            f"Best energy in SQD iteration {round_idx}: "
            f"{best_result_in_batch.energy + elec_props.nuclear_repulsion_energy} Ha."
        )

    # Upload artifact
    await create_table_artifact(
        table=sqd_artifact,
        key="sqd-configuration-recovery",
    )
    return best_result.energy + elec_props.nuclear_repulsion_energy


@task(
    persist_result=True,
    result_serializer="compressed/json",
)
def compute_molecular_integrals(
    mol_params: MoleculeGeometry | FCIDumpFile,
) -> ElectronicProperties:
    """Precompute molecular orbital property with classical methods."""
    
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    if isinstance(mol_params, MoleculeGeometry):
        mol = gto.Mole()
        mol.build(
            atom=mol_params.atom,
            basis=mol_params.basis,
            symmetry=mol_params.symmetry,
        )
        mol.stdout = buf
        mol.verbose = 4
        mf = scf.RHF(mol).run()
        norb = mf.mo_coeff.shape[1]

        # AO integrals
        hcore = mf.get_hcore()
        eri = mol.intor("int2e")
        spin_sq = mol_params.spin_sq

    elif isinstance(mol_params, FCIDumpFile):
        data = tools.fcidump.read(mol_params.fcidump_file)
        norb = data["NORB"]
        
        mf = tools.fcidump.to_scf(mol_params.fcidump_file)
        mf.mol.verbose = 4
        mf.mol.stdout = buf

        # Run HF calculation with Newton method.
        # HF convergence is important, as we assume 
        # the FCIdump file is created with a converged result.
        mf = scf.newton(mf)
        mf.symmetry = False
        dm0 = np.zeros((norb, norb))
        for i in range(mf.mol.nelectron // 2):
            dm0[i,i] = 2.0
        mf.kernel(dm0=dm0)
        
        # MO integrals. These are raw Hamiltonian.
        hcore = mf.get_hcore()
        eri = ao2mo.restore(1, mf._eri, norb)        
        spin_sq = mol_params.spin_sq

    else:
        raise TypeError("Unreachable.")

    # Apply unitary transform with obtained MO coefficient.
    # The FCIdump file already gives you these integrals in MO basis, 
    # but the HF calculation may give you correction to these integrals.
    # For example, phase may change with this unitary transform.
    # When started with Mole object, AO → MO transform is performed in here.
    h1 = mf.mo_coeff.T @ hcore @ mf.mo_coeff
    h2 = ao2mo.full(eri, mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)
    
    nuclear_repulsion_energy = mf.mol.energy_nuc()
    num_elec_a, num_elec_b = mf.mol.nelec

    # Run CCSD
    mycc = cc.CCSD(mf)
    mycc.kernel()
    t2 = mycc.t2
    
    # Diagonalize RDM to obtain the occupancies
    # Eigenvectors are the natual molecular orbitals
    rdm1_ccsd = mf.make_rdm1()
    occ_ccsd, _ = scipy.linalg.eigh(rdm1_ccsd)
    occ_ccsd /= 2.0

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())
    
    return ElectronicProperties(
        one_body_tensor=h1,
        two_body_tensor=h2,
        t2=t2,
        initial_occupancy=(occ_ccsd[::-1], occ_ccsd[::-1]),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        spin_sq=spin_sq,
    )


@task(
    persist_result=True,
    result_serializer="compressed/pickle",
)
async def sample_bitstrings(
    circuit_params: CircuitParameters,
    elec_props: ElectronicProperties,
    runner_name: str,
    option_name: str,
) -> BitArray:
    """Sample bitstring with quantum computer.

    Args:
        circuit_params: Flow parameters to construct ansatz quantum circuit.
        elec_props: Electronic properties of molecule to solver for.
        runner_name: Block name of QuantumRuntime to execute primitives on.
        option_name: Name of Variable storing sampler primitive options to load.

    Returns:
        Sampled bitstrings representing determinants.
    """
    logger = get_run_logger()

    try:
        # Run on a real hardware when credentials are found.
        runtime = await QuantumRuntime.load(runner_name)
        options = await Variable.get(option_name)
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

    # Create ansatz circuits
    logger.info("Creating ansatz circuit...")
    alpha_alpha_indices = [(p, p + 1) for p in range(elec_props.num_orbitals - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, elec_props.num_orbitals, 4)]
    interaction_pairs = alpha_alpha_indices, alpha_beta_indices

    qreg = QuantumRegister(2 * elec_props.num_orbitals, name="q")
    creg_test = ClassicalRegister(2 * elec_props.num_orbitals, name="test")
    creg_meas = ClassicalRegister(2 * elec_props.num_orbitals, name="meas")

    regs = [qreg, creg_meas]
    if circuit_params.use_reset_mitigation:
        regs.append(creg_test)
        
    lucj_circ = QuantumCircuit(*regs)
    if circuit_params.use_reset_mitigation:
        lucj_circ.measure(qreg, creg_test)
        lucj_circ.barrier()
    lucj_circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        ),
        qargs=qreg,
    )
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=elec_props.t2,
        n_reps=circuit_params.n_lucj_layers,
        interaction_pairs=interaction_pairs,
    )
    lucj_circ.append(
        ffsim.qiskit.UCJOpSpinBalancedJW(
            ucj_op=ucj_op,
        ),
        qargs=qreg,
    )
    lucj_circ.measure(qreg, creg_meas)

    # Transpile
    target = await runtime.get_target()
    coupling_map = target.build_coupling_map()

    passmanager = generate_preset_pass_manager(
        optimization_level=circuit_params.optimization_level,
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

    logger.info("Transpiling ansatz circuit...")
    isa_circuit = passmanager.run(lucj_circ)
    gate_depth = isa_circuit.depth(
        lambda inst: inst.operation.name not in ("rz", "barrier", "measure")
    )
    logger.info(
        f"Circuit depth = {gate_depth}\n"
        f"Instruction counts = {dict(isa_circuit.count_ops())}\n"
    )

    # Run primitive
    pub_result = await runtime.sampler(
        sampler_pubs=[(isa_circuit,)],
        options=options,
    )

    # Post-process bitstrings
    meas_bits = pub_result[0].data.meas
    if circuit_params.use_reset_mitigation:
        test_bits = pub_result[0].data.test
        bit_array = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
        bit_array = BitArray.from_samples(bit_array, num_bits=meas_bits.num_bits)
        logger.info(
            "Reset mitigation result:\n"
            f"  Before: {meas_bits.num_shots} bitstrings\n"
            f"  After: {bit_array.num_shots} bitstrings\n"
            f"  Retention rate: {bit_array.num_shots / meas_bits.num_shots}\n"
        )
    else:
        bit_array = meas_bits

    return bit_array


def deploy():
    """Deploy workflow with a local worker."""
    import os
    import pathlib

    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    sqd_2405_05068.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="sqd_2405_05068",
        description="SQD experiment from arXiv2405.05068.",
    )
