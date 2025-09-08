"""Definition of SQD workflow."""

import asyncio

import ffsim
import numpy as np
import pyscf
import pyscf.mcscf
from ffsim.qiskit import PRE_INIT
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
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from prefect_dice import DiceSHCISolverJob

from .subsample import postselect, subsample


class Parameters(BaseModel):
    """Workflow parameters to configure sqd_2405_05068 workflow."""

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
    sqd_subspace_dim: int = Field(
        description="SQD: Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )
    sqd_num_batches: int = Field(
        description="SQD: Number of batches of configurations used by the different calls to the eigenstate solver.",
        title="Batch Number",
        ge=1,
    )
    sqd_max_iterations: int = Field(
        description="SQD: Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
    )
    n_lucj_layers: int = Field(
        default=1,
        description="Number of repetition of the unit layer of LUCJ ansatz.",
        title="LUCJ Layers",
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
        default=10000,
        description="Transpile: The number of random initial layouts tested, selecting the one that minimizes SWAP gates.",
        title="SABRE Layout Trials",
        ge=1,
    )


@flow
async def sqd_2405_05068(
    parameters: Parameters,
    runner_name: str = "sqd-runner",
    solver_name: str = "sqd-solver",
):
    """SQD Experiment from arXiv2405.05068."""
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

        ansatz_circuit = create_ansatz_circuits(
            parameters=parameters,
            one_body_tensor=hcore,
            num_electrons=nelec,
            t2=t2,
        )
        isa_circuit = transpile_circuit(
            circuit=ansatz_circuit,
            target=await runtime.get_target(),
            parameters=parameters,
            seed=rng,
        )
        pub_result = await runtime.sampler(
            sampler_pubs=[(isa_circuit,)],
            options=options,
        )
        bit_array = pub_result[0].data.meas
    except ValueError:
        # Uniform sampling when runtime is not defined.
        logger.warning(
            f"Runner block {runner_name} is not defined. "
            "Falling back into random uniform sampling."
        )
        bit_array = generate_bit_array_uniform(10_000, norb * 2, rand_seed=rng)

    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    n_alpha, n_beta = nelec

    # Run configuration recovery loop
    sci_solver = await DiceSHCISolverJob.load(solver_name)
    sqd_artifact = []
    current_occupancies = None
    best_result = None
    for round_idx in range(parameters.sqd_max_iterations):
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
            subspace_dim=parameters.sqd_subspace_dim,
            num_batches=parameters.sqd_num_batches,
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
            sqd_artifact.append(new_record)
        logger.info(
            f"Best energy in SQD iteration {round_idx}: {best_result_in_batch.energy + nuclear_repulsion_energy} Ha."
        )

    # Upload artifact
    await create_table_artifact(
        table=sqd_artifact,
        key="sqd-configuration-recovery",
    )
    return best_result.energy + nuclear_repulsion_energy


@task
def compute_molecular_integrals(
    parametrers: Parameters,
) -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int]]:
    """Compute molecular property with the classical method."""
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
def create_ansatz_circuits(
    parameters: Parameters,
    one_body_tensor: np.ndarray,
    num_electrons: tuple[int, int],
    t2: np.ndarray,
) -> QuantumCircuit:
    """Create LUCJ ansatz circuit with T2 parameter."""
    num_orbitals = one_body_tensor.shape[0]

    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, num_orbitals, 4)]
    interaction_pairs = alpha_alpha_indices, alpha_beta_indices

    qreg = QuantumRegister(2 * num_orbitals, name="q")
    circ = QuantumCircuit(qreg)
    circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=num_orbitals,
            nelec=num_electrons,
        ),
        qargs=qreg,
    )
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        n_reps=parameters.n_lucj_layers,
        interaction_pairs=interaction_pairs,
    )
    circ.append(
        ffsim.qiskit.UCJOpSpinBalancedJW(
            ucj_op=ucj_op,
        ),
        qargs=qreg,
    )
    circ.measure_all()

    return circ


@task
def transpile_circuit(
    circuit: QuantumCircuit,
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
    isa_circuit = passmanager.run(circuit)
    gate_depth = isa_circuit.depth(
        lambda inst: inst.operation.name not in ("rz", "barrier", "measure")
    )
    logger.info(
        f"Circuit depth = {gate_depth}\n"
        f"Instruction counts = {dict(isa_circuit.count_ops())}¥n"
    )
    return isa_circuit


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
