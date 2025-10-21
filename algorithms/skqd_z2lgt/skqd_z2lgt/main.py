"""Definition of the main Prefect flow for Z2LGT SKQD."""

import os
from pathlib import Path
import logging
import asyncio
import tempfile
import numpy as np
import h5py
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
from qiskit.primitives import BitArray
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
from prefect import flow, task, get_run_logger
from prefect.variables import Variable
from pydantic import BaseModel, Field
from prefect_qiskit.runtime import QuantumRuntime
from prefect_qiskit.primitives import PrimitiveJobRun
from prefect_miyabi import MiyabiJobBlock, PyFunctionJob
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits
from skqd_z2lgt.recovery_learning import preprocess

TASK_SCRIPT_DIR = Path(__file__).parents[0] / 'tasks'


class LGTParameters(BaseModel):
    """Parameters to specify the Z2 LGT."""

    lattice: str = Field(
        default='full_156q',
        description='Two-dimensional lattice configuration.',
        title='Lattice Configuration'
    )
    plaquette_energy: float = Field(
        default=1.,
        description='Plaquette energy in the Hamiltonian (transverse field strength in the dual'
                    ' Ising model).',
        title='Plaquette Energy',
        ge=0.
    )

    def model_post_init(self, _):  # pylint: disable=arguments-differ
        if self.lattice == 'full_156q':
            self.lattice = '''
            *-*-*-*-*╷
             * * * * *
            * * * * *╎
             * * * * *
            * * * * *╎
             * * * * *
            * * * * *╎
             * * * * *
            *-*-*-*-*╵
            '''


class CircuitParameters(BaseModel):
    """Configuration for circuit."""

    basis_2q: str = Field(
        default='rzz',
        description='Two-qubit gate used in the Trotter circuit ("cx", "cz", or "rzz"). Note that'
                    ' the selection "rzz" will utilize CZ gates together with Rzz.',
        title='Base two-qubit gate'
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )


class CRBMParameters(BaseModel):
    """Configuration for conditional restricted Boltzmann machine used in configuration recovery."""

    num_h: int = Field(
        default=256,
        description='Number of hidden units of the restricted Boltzmann machine.',
        title='Number of Hidden Units',
        ge=1
    )
    l2w_weights: float = Field(
        default=1.,
        description='Weight of the L2 regularization term for weight parameters of the model.',
        title='L2 Weight (W)',
        ge=0.
    )
    l2w_biases: float = Field(
        default=0.2,
        description='Weight of the L2 regularization term for bias parameters of the model.',
        title='L2 Weight (B)',
        ge=0.
    )
    batch_size: int = Field(
        default=32,
        description='Batch size for stochastic gradient descent.',
        title='Training Batch Size',
        ge=1
    )
    learning_rate: float = Field(
        default=0.001,
        description='Learning rate for the Adam optimizer with weight decay',
        title='Learning Rate',
        gt=0.
    )
    init_h_sparsity: float = Field(
        default=0.01,
        description='Sparsity parameter for initializing the bias parameters of the hidden units.',
        title='Initial Hidden Unit Sparsity',
        ge=0.
    )
    num_epochs: int = Field(
        default=10,
        description='Maximum number of epochs to train.',
        title='Maximum Epochs',
        ge=1
    )
    rtol: float = Field(
        default=2.,
        description='Convergence condition (max|loss - mean| of the last 5 epochs < rtol * stddev)',
        title='Convergence Relative Tolerance',
        ge=0.
    )


class SKQDParameters(BaseModel):
    """Configuration for SKQD algorithm execution."""

    n_trotter_steps: int = Field(
        default=8,
        description="Number of Trotter steps (Krylov dimension).",
        title="Trotter Steps",
        ge=1,
    )
    dt: float = Field(
        default=0.02,
        description="Time Interval for Trotterization.",
        title="Time Interval",
        ge=0,
    )
    max_subspace_dim: int = Field(
        default=1_000_000,
        description="Maximum subspace dimension in iterative configuration recovery.",
        title="Maximum Subspace Dimension",
        ge=1
    )
    num_gen: int = Field(
        default=3,
        description="Number of bitstrings to generate in each iteration of configuration recovery.",
        title="Generated Samples",
        ge=1
    )
    max_iterations: int = Field(
        default=10,
        description="Number of configuration recovery iterations.",
        title="Max Iteration",
        ge=0
    )
    delta_e: float = Field(
        default=0.005,
        description="Convergence condition (change in energy value between configuration recovery"
                    "iterations)",
        title="Energy Convergence",
        ge=0.
    )


class Parameters(BaseModel):
    """Workflow configuration parameters."""

    lgt: LGTParameters = Field(
        default_factory=LGTParameters,
        description='Lattice gauge theory definition.',
        title='Lattice Gauge Theory'
    )

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description='Trotter circuit parameters and transpiler settings.',
        title='Circuit'
    )

    crbm: CRBMParameters = Field(
        default_factory=CRBMParameters,
        description='Settings for conditional restricted Boltzmann machine.',
        title='CRBM'
    )

    skqd: SKQDParameters = Field(
        default_factory=SKQDParameters,
        description='Control of SKQD algorithm execution and solver parameters.',
        title='SKQD',
    )

    runtime_job_id: str | None = Field(
        default=None,
        description='ID of an existing IBM Quantum workload.',
        title='Runtime Job ID'
    )

    output_filename: str | None = Field(
        default=None,
        description='Name of the HDF5 file where intermediate output are stored.',
        title='Output File Name'
    )


@flow
async def skqd_z2lgt(
    parameters: Parameters,
    runner_name: str = 'ibm-runner',
    option_name: str = 'sampler_options',
    cpu_pyfuncjob_name: str = 'cpu-pyfunc',
    cuda_scriptjob_name: str = 'cuda-script'
) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD.

    Args:
        parameters: Configuration parameters.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable for QuantumRunner sampler options.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
    """
    logger = get_run_logger()
    logger.setLevel(logging.INFO)

    output_filename = open_output(parameters)
    logger.info('Running a quantum job to obtain the bitstrings')
    bit_arrays = await sample_krylov_bitstrings(parameters, runner_name, option_name,
                                                output_filename)

    logger.info('Correcting and converting link states to plaquette states')
    await preprocess_bitstrings(parameters, cpu_pyfuncjob_name, bit_arrays, output_filename)
    logger.info('Training conditional restricted Boltzmann machines')
    await train_crbm(parameters, cuda_scriptjob_name, output_filename)
    logger.info('Performing SQD with configuration recovery')
    await project_and_diagonalize(parameters, cuda_scriptjob_name, output_filename)

    with h5py.File(output_filename) as source:
        energy = source['skqd_rcv/energy'][()]
        eigvec = source['skqd_rcv/eigvec'][()]

    logger.info('Estimated ground-state energy is %f', energy)

    if not parameters.output_filename:
        os.unlink(output_filename)

    return energy, eigvec


@task
def open_output(parameters: Parameters) -> str:
    """Open a new output HDF5 file and set it up for the workflow, or validate an existing file.

    Args:
        parameters: Workflow parameters.

    Returns:
        The name of the output file.
    """
    logger = get_run_logger()

    output_filename = parameters.output_filename
    if not output_filename:
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            output_filename = tfile.name

    try:
        with h5py.File(output_filename, 'r'):
            pass
        logger.info('Using existing file %s', output_filename)
    except FileNotFoundError:
        logger.info('Creating a new file %s', output_filename)
        with h5py.File(output_filename, 'w', libver='latest'):
            pass

    return output_filename


@task
def get_trotter_circuits(
    parameters: Parameters,
    target: Target
) -> tuple[list[int], list[QuantumCircuit], list[QuantumCircuit]]:
    """Compose full Trotter simulation circuits.

    We first generate single-step circuit elements for the given lattice and base two-qubit gate,
    compile them, then compose the resulting ISA circuits into multi-step Trotter simulation
    circuits. Both the forward-evolution (Krylov) circuits and forward-backward (reference) circuits
    are returned.

    Args:
        parameters: Workflow parameters.
        target: Backend target.

    Returns:
        Physical qubit layout and lists (length configuration.num_steps) of forward-evolution and
        forward-backward circuits.
    """
    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    layout = lattice.layout_heavy_hex(target=target, basis_2q=parameters.circuit.basis_2q)

    pm = generate_preset_pass_manager(
        optimization_level=parameters.circuit.optimization_level,
        target=target,
        initial_layout=layout,
    )
    pm.post_optimization = PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
            RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
        ]
    )
    circuits = make_step_circuits(lattice, parameters.lgt.plaquette_energy,
                                  parameters.skqd.dt, parameters.circuit.basis_2q)
    # Somehow the combination of multiprocessing pm.run + prefect task causes the former to hang
    full_step, fwd_step, bkd_step, measure = [pm.run(circuit) for circuit in circuits]

    id_step = fwd_step.compose(bkd_step)
    exp_circuits = compose_trotter_circuits(full_step, measure, parameters.skqd.n_trotter_steps)
    ref_circuits = compose_trotter_circuits(id_step, measure, parameters.skqd.n_trotter_steps)
    return layout, exp_circuits, ref_circuits


@task
async def sample_krylov_bitstrings(
    parameters: Parameters,
    runner_name: str,
    option_name: str,
    output_filename: str
) -> tuple[list[BitArray], list[BitArray]]:
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable storing sampler primitive options.
        output_filename: Name of the HDF5 file where intermediate and final output of the workflow
            are written.

    Returns:
        Lists of BitArrays for forward-evolution and forward-backward circuits.
    """
    logger = get_run_logger()
    num_steps = parameters.skqd.n_trotter_steps

    with h5py.File(output_filename, 'r') as source:
        try:
            group = source['data/raw']
        except KeyError:
            pass
        else:
            logger.info('Loading existing raw data from output file')
            dlists = ([], [])
            for etype, dlist in zip(['exp', 'ref'], dlists):
                for istep in range(num_steps):
                    dataset = group[f'{etype}_step{istep}']
                    dlist.append(BitArray(dataset[()], int(dataset.attrs['num_bits'])))

            return dlists

    runtime = await QuantumRuntime.load(runner_name)

    if parameters.runtime_job_id:
        logger.info('Fetching result of workload %s', parameters.runtime_job_id)
        job = PrimitiveJobRun(job_id=parameters.runtime_job_id, credentials=runtime.credentials)
        pub_result = await job.fetch_result()
        layout = None
    else:
        options = await Variable.get(option_name)

        # Transpile and compose the circuits
        target = await runtime.get_target()
        layout, exp_circuits, ref_circuits = get_trotter_circuits(parameters, target)

        # Run primitive
        pub_result = await runtime.sampler(
            sampler_pubs=exp_circuits + ref_circuits,
            options=options,
        )

    with h5py.File(output_filename, 'r+') as out:
        try:
            del out['data/raw']
        except KeyError:
            pass
        group = out.create_group('data/raw')
        if layout:
            group.attrs['layout'] = np.array(layout)
        for ires, res in enumerate(pub_result):
            if ires < num_steps:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = ires % num_steps
            bit_array = res.data.c
            dataset = group.create_dataset(f'{etype}_step{istep}', data=bit_array.array)
            dataset.attrs['num_bits'] = bit_array.num_bits

    return ([res.data.c for res in pub_result[:num_steps]],
            [res.data.c for res in pub_result[num_steps:]])


@task
async def preprocess_bitstrings(
    parameters: Parameters,
    cpu_pyfuncjob_name: str,
    bit_arrays: tuple[list[BitArray], list[BitArray]],
    output_filename: str
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        bit_arrays: Lists of BitArrays returned by sample_krylov_bitstrings.
        output_filename: Name of the HDF5 file where intermediate and final output of the workflow
            are written.
    """
    logger = get_run_logger()

    with h5py.File(output_filename, 'r') as source:
        data_group = source.get('data', {})
        if 'vtx' in data_group and 'plaq' in data_group:
            logger.info('Using existing vertex and plaquette data')
            return

    job_block = await PyFunctionJob.load(cpu_pyfuncjob_name)
    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    dual_lattice = lattice.plaquette_dual()
    batch_size = bit_arrays[0][0].array.shape[0] // 20

    tasks = []
    async with asyncio.TaskGroup() as taskgroup:
        for bit_array in bit_arrays[0] + bit_arrays[1]:
            tasks.append(
                taskgroup.create_task(
                    job_block.run(preprocess, bit_array, dual_lattice, batch_size=batch_size)
                )
            )

    with h5py.File(output_filename, 'r+') as out:
        lengths = [lattice.num_vertices, lattice.num_plaquettes]
        data_group = out['data']
        groups = [data_group.get(gname) or data_group.create_group(gname)
                  for gname in ['vtx', 'plaq']]

        for idx, atask in enumerate(tasks):
            arrays = atask.result()
            if idx < parameters.skqd.n_trotter_steps:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = idx % parameters.skqd.n_trotter_steps
            dname = f'{etype}_step{istep}'
            for group, array, num_bits in zip(groups, arrays, lengths):
                try:
                    del group[dname]
                except KeyError:
                    pass
                dataset = group.create_dataset(dname, data=np.packbits(array, axis=1))
                dataset.attrs['num_bits'] = num_bits


@task
async def train_crbm(
    parameters: Parameters,
    cuda_scriptjob_name: str,
    output_filename: str
):
    """Train a CRBM per Trotter step.

    Args:
        parameters: Configuration parameters.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
        output_filename: Name of the HDF5 file where intermediate and final output of the workflow
            are written.
    """
    logger = get_run_logger()

    steps = []
    with h5py.File(output_filename, 'r+') as out:
        group = out.get('crbm') or out.create_group('crbm')
        num_steps = int(out.attrs['num_steps'])
        for istep in range(num_steps):
            if f'step{istep}' not in group:
                steps.append(istep)

    if not steps:
        logger.info('All models already trained')
        return

    logger.info('Training CRBMs for Trotter steps %s', steps)

    conf = parameters.crbm

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    with job_block.get_executor() as executor:
        tasks = []
        async with asyncio.TaskGroup() as taskgroup:
            for istep in steps:
                tempname = Path(executor.work_dir) / f'crbm_step{istep}.h5'
                arguments = [
                    TASK_SCRIPT_DIR / 'train_crbm.py',
                    output_filename,
                    f'{istep}',
                    '--out-filename', tempname,
                    '--num-h', f'{conf.num_h}',
                    '--l2w-weights', f'{conf.l2w_weights}',
                    '--l2w-biases', f'{conf.l2w_biases}',
                    '--init-h-sparsity', f'{conf.init_h_sparsity}',
                    '--batch-size', f'{conf.batch_size}',
                    '--learning-rate', f'{conf.learning_rate}',
                    '--num-epochs', f'{conf.num_epochs}',
                    '--rtol', f'{conf.rtol}'
                ]
                atask = taskgroup.create_task(
                    executor.execute_job(
                        arguments=arguments,
                        **job_block.get_job_variables()
                    )
                )
                tasks.append((istep, atask, tempname))

        with h5py.File(output_filename, 'r+') as out:
            group = out['crbm']
            for istep, atask, tempname in tasks:
                if (code := atask.result()) != 0:
                    raise RuntimeError(f'CRBM training return code {code} for Trotter step {istep}')

                with h5py.File(tempname, 'r') as source:
                    source.copy(f'crbm/step{istep}', group)


@task
async def project_and_diagonalize(
    parameters: Parameters,
    cuda_scriptjob_name: str,
    output_filename: str
):
    """Perform SQD with iterative configuration recovery.

    Args:
        parameters: Configuration parameters.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
        output_filename: Name of the HDF5 file where intermediate and final output of the workflow
            are written.
    """
    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    with job_block.get_executor() as executor:
        arguments = [
            TASK_SCRIPT_DIR / 'skqd_recovery.py',
            output_filename,
            '--gpu', 'all',
            '--num-gen', f'{parameters.skqd.num_gen}',
            '--niter', f'{parameters.skqd.max_iterations}',
            '--terminate', f'diff={parameters.skqd.delta_e}',
            f'dim={parameters.skqd.max_subspace_dim}'
        ]
        await executor.execute_job(
            arguments=arguments,
            **job_block.get_job_variables()
        )


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(Path(__file__).parent)

    # Deploy the workflow with specified options.
    skqd_z2lgt.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="skqd_z2lgt",
        description="SKQD experiment for Z2 LGT."
    )
