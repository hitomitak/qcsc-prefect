"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
import h5py
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit.primitives import BitArray
from prefect import flow, task, get_run_logger
from prefect.variables import Variable
from prefect_qiskit.runtime import QuantumRuntime
from prefect_qiskit.primitives import PrimitiveJobRun
from prefect_miyabi import MiyabiJobBlock, PyFunctionJob
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.mwpm import convert_link_to_plaq, minimum_weight_link_state
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.open_output import open_output as _open_output
from skqd_z2lgt.tasks.sample_quantum import sample_quantum_flow


TASK_SCRIPT_DIR = Path(__file__).parents[0] / 'tasks'


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
    bit_arrays = await sample_quantum(parameters, runner_name, option_name, output_filename)

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
    return _open_output(parameters, get_run_logger())


@task
async def sample_quantum(
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
    async with asyncio.TaskGroup() as tg:
        runtime_task = tg.create_task(QuantumRuntime.load(runner_name))
        options_task = tg.create_task(Variable.get(option_name))

    runtime = runtime_task.result()
    options = options_task.result()

    def fetch_result_fn():
        def fn():
            job = PrimitiveJobRun(job_id=parameters.runtime_job_id, credentials=runtime.credentials)
            return asyncio.run(job.fetch_result())

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    def get_target_fn():
        def fn():
            return asyncio.run(runtime.get_target())

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    def sample_fn(pubs):
        def fn(pubs):
            return asyncio.run(runtime.sampler(sampler_pubs=pubs, options=options))

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn, pubs).result()

    return sample_quantum_flow(parameters, output_filename, fetch_result_fn, get_target_fn,
                               sample_fn, logger)


def convert_bit_arrays(bit_arrays, dual_lattice, batch_size):
    return [convert_link_to_plaq(bit_array, dual_lattice, batch_size) for bit_array in bit_arrays]


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
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)
    batch_size = bit_arrays[0][0].array.shape[0] // 20

    tasks = []
    async with asyncio.TaskGroup() as taskgroup:
        for arrays in bit_arrays:
            tasks.append(taskgroup.create_task(
                job_block.run(convert_bit_arrays, arrays, dual_lattice, batch_size)
            ))

    with h5py.File(output_filename, 'r+') as out:
        lengths = [lattice.num_vertices, lattice.num_plaquettes]
        data_group = out['data']
        groups = [data_group.get(gname) or data_group.create_group(gname)
                  for gname in ['vtx', 'plaq']]

        for etype, atask in zip(['exp', 'ref'], tasks):
            for istep, arrays in enumerate(atask.result()):
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
        for istep in range(parameters.skqd.n_trotter_steps):
            if f'step{istep}' not in group:
                steps.append(istep)

    if not steps:
        logger.info('All models already trained')
        return

    logger.info('Training CRBMs for Trotter steps %s', steps)

    conf = parameters.crbm

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)

    async def run_train_job(istep, data_dir):
        with job_block.get_executor() as executor:
            arguments = [
                TASK_SCRIPT_DIR / 'train_crbm.py',
                output_filename,
                f'{istep}',
                '--out-filename', data_dir / 'out.h5',
                '--num-h', f'{conf.num_h}',
                '--l2w-weights', f'{conf.l2w_weights}',
                '--l2w-biases', f'{conf.l2w_biases}',
                '--init-h-sparsity', f'{conf.init_h_sparsity}',
                '--batch-size', f'{conf.batch_size}',
                '--learning-rate', f'{conf.learning_rate}',
                '--num-epochs', f'{conf.num_epochs}',
                '--rtol', f'{conf.rtol}'
            ]
            return await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )

    tasks = []
    async with asyncio.TaskGroup() as taskgroup:
        for istep in steps:
            data_dir = Path(tempfile.mkdtemp(prefix='data_', dir=job_block.work_root))
            logger.info('Trained model for step %d will be written to %s', istep, data_dir)
            atask = taskgroup.create_task(run_train_job(istep, data_dir))
            tasks.append((istep, atask, data_dir))

    with h5py.File(output_filename, 'r+') as out:
        group = out['crbm']
        for istep, atask, data_dir in tasks:
            if (code := atask.result()) != 0:
                raise RuntimeError(f'CRBM training return code {code} for Trotter step {istep}')

            with h5py.File(data_dir / 'out.h5', 'r') as source:
                source.copy(f'crbm/step{istep}', group)

            shutil.rmtree(data_dir)


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
    logger = get_run_logger()

    with h5py.File(output_filename, 'r') as source:
        if 'skqd_rcv' in source:
            logger.info('SQD result already exists')
            return

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
