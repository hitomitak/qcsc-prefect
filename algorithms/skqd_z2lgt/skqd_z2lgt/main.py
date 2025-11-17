"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import h5py
from prefect import flow, task, get_run_logger
from prefect.variables import Variable
from prefect_qiskit.runtime import QuantumRuntime
from prefect_qiskit.primitives import PrimitiveJobRun
from prefect_miyabi import MiyabiJobBlock, PyFunctionJob
from skqd_z2lgt.ising_dmrg import ising_dmrg, get_mps_probs
from skqd_z2lgt.mwpm import convert_link_to_plaq
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.open_output import open_output as _open_output
from skqd_z2lgt.tasks.dmrg import dmrg_flow
from skqd_z2lgt.tasks.sample_quantum import sample_quantum_flow, load_raw
from skqd_z2lgt.tasks.preprocess import preprocess_flow
from skqd_z2lgt.tasks.train_generator import train_generator_flow, load_model, save_model


TASK_SCRIPT_DIR = Path(__file__).parents[0] / 'tasks'


@flow
async def skqd_z2lgt(
    parameters: Parameters,
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

    tmpdir = None
    if not parameters.pkgpath:
        tmpdir = tempfile.TemporaryDirectory()
        parameters.pkgpath = tmpdir.name

    open_output(parameters)
    if parameters.dmrg:
        logger.info('Estimating ground-state energy via DMRG')
        dmrg_energy = await dmrg(parameters, cpu_pyfuncjob_name)
    logger.info('Running a quantum job to obtain the bitstrings')
    raw_data = await sample_quantum(parameters)
    logger.info('Correcting and converting link states to plaquette states')
    reco_data = await preprocess(parameters, raw_data, cpu_pyfuncjob_name)
    logger.info('Training conditional restricted Boltzmann machines')
    crbm_models = await train_generator(parameters, reco_data[1], cuda_scriptjob_name)
    logger.info('Performing SQD with random bit flips')
    energy_random, _ = await diagonalize(parameters, reco_data[0], None,
                                         cuda_scriptjob_name=cuda_scriptjob_name)
    logger.info('Performing SQD with configuration recovery')
    energy_norecov, energy = await diagonalize(parameters, reco_data[0], crbm_models,
                                               cuda_scriptjob_name=cuda_scriptjob_name)

    if tmpdir:
        tmpdir.cleanup()

    if parameters.dmrg:
        logger.info('DMRG energy: %f', dmrg_energy)
    logger.info('SKQD energy (no conf. recovery): %f', energy_norecov)
    logger.info('DMRG energy (random bit flips): %f', energy_random)
    logger.info('DMRG energy: %f', energy)
    logger.info('Estimated ground-state energy is %f', energy)

    return energy


@task
def open_output(parameters: Parameters):
    return _open_output(parameters, get_run_logger())


@task
async def dmrg(
    parameters: Parameters,
    cpu_pyfuncjob_name: str
) -> float:
    """Run DMRG and MPS sampling."""
    logger = get_run_logger()

    job_block = await PyFunctionJob.load(cpu_pyfuncjob_name)

    dmrg_params = parameters.dmrg
    julia_bin = 'julia'
    if dmrg_params.julia_sysimage:
        julia_bin = ['julia', '--sysimage', dmrg_params.julia_sysimage]

    def dmrg_fn(hamiltonian, filename):
        async def fn():
            return await job_block.run(ising_dmrg, hamiltonian, filename=filename,
                                       num_sweeps=dmrg_params.num_sweeps,
                                       maxdim=dmrg_params.maxdim, cutoff=dmrg_params.cutoff,
                                       julia_bin=julia_bin)

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(fn())).result()

    def mps_probs_fn(filename):
        async def fn():
            return await job_block.run(get_mps_probs, filename, num_samples=dmrg_params.num_samples,
                                       julia_bin=julia_bin)

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(fn())).result()

    return dmrg_flow(parameters, dmrg_fn, mps_probs_fn, logger)


@task
async def sample_quantum(
    parameters: Parameters
) -> tuple[None, None]:
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable storing sampler primitive options.
    """
    logger = get_run_logger()
    async with asyncio.TaskGroup() as tg:
        runtime_task = tg.create_task(QuantumRuntime.load(parameters.runtime.runtime_block_name))
        options_task = tg.create_task(Variable.get(parameters.runtime.options_name))

    runtime = runtime_task.result()
    options = options_task.result()

    def fetch_result_fn():
        def fn():
            job = PrimitiveJobRun(job_id=parameters.runtime.job_id, credentials=runtime.credentials)
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

    sample_quantum_flow(parameters, fetch_result_fn, get_target_fn, sample_fn, logger)
    # As a flow, this function should return a tuple of raw data. We however do not need these
    # large arrays to be resident on memory of the scheduler job, so will instead just return a
    # dummy object.
    return None, None


def convert_bit_arrays(parameters, etype, dual_lattice):
    bit_arrays = load_raw(parameters, etype)
    batch_size = parameters.runtime.shots // 20
    return [convert_link_to_plaq(bit_array, dual_lattice, batch_size) for bit_array in bit_arrays]


@task
async def preprocess(
    parameters: Parameters,
    _: tuple[None, None],
    cpu_pyfuncjob_name: str
) -> tuple[None, None]:
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
    """
    logger = get_run_logger()

    def convert_fn(_, dual_lattice):
        async def fn():
            # Better to have a dedicated self-contained block that loads and saves data
            job_block = await PyFunctionJob.load(cpu_pyfuncjob_name)
            tasks = []
            async with asyncio.TaskGroup() as taskgroup:
                for etype in ['exp', 'ref']:
                    tasks.append(taskgroup.create_task(
                        job_block.run(convert_bit_arrays, parameters, etype, dual_lattice)
                    ))
            return tuple(atask.result() for atask in tasks)

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(fn())).result()

    preprocess_flow(parameters, None, convert_fn, logger)
    # See comment in sample_quantum
    return None, None


@task
async def train_generator(
    parameters: Parameters,
    _: None,
    cuda_scriptjob_name: str
):
    """Train a CRBM per Trotter step.

    Args:
        parameters: Configuration parameters.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
        pkgpath: Name of the HDF5 file where intermediate and final output of the workflow
            are written.
    """
    logger = get_run_logger()

    conf = parameters.crbm
    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)

    async def run_train_job(istep, data_dir):
        with job_block.get_executor() as executor:
            arguments = [
                TASK_SCRIPT_DIR / 'train_generator.py',
                parameters.pkgpath,
                f'{istep}',
                '--out-filename', data_dir / 'out.h5',
                '--num-h', f'{conf.num_h}',
                '--l2w-weights', f'{conf.l2w_weights}',
                '--l2w-biases', f'{conf.l2w_biases}',
                '--init-h-sparsity', f'{conf.init_h_sparsity}',
                '--batch-size', f'{conf.train_batch_size}',
                '--learning-rate', f'{conf.learning_rate}',
                '--num-epochs', f'{conf.num_epochs}',
                '--rtol', f'{conf.rtol}'
            ]
            return await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )

    async def run_train_jobs(steps_to_train):
        tasks = []
        async with asyncio.TaskGroup() as taskgroup:
            for istep in steps_to_train:
                data_dir = Path(tempfile.mkdtemp(prefix='data_', dir=job_block.work_root))
                logger.info('Trained model for step %d will be written to %s', istep, data_dir)
                atask = taskgroup.create_task(run_train_job(istep, data_dir))
                tasks.append((istep, atask, data_dir))

        models = []
        for istep, atask, data_dir in tasks:
            if (code := atask.result()) != 0:
                raise RuntimeError(f'CRBM training return code {code} for Trotter step {istep}')
            model, records = load_model(istep, data_dir / 'out.h5')
            save_model(istep, model, records, parameters.pkgpath)
            shutil.rmtree(data_dir)
            models.append(model)

        return models

    def train_fn(steps_to_train, _):
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_train_jobs(steps_to_train))).result()

    train_generator_flow(parameters, None, train_fn, logger)
    return [None] * parameters.skqd.n_trotter_steps


@task
async def diagonalize(
    parameters: Parameters,
    _data: None,
    crbm_models: list[None] | None,
    cuda_scriptjob_name: str
) -> tuple[float, float]:
    """Perform SQD with iterative configuration recovery.

    Args:
        parameters: Configuration parameters.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
    """
    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    with job_block.get_executor() as executor:
        arguments = [
            TASK_SCRIPT_DIR / 'diagonalize.py',
            parameters.pkgpath,
            '--gpu', 'all'
        ]
        if isinstance(crbm_models, list):
            arguments += ['--mode', 'full']
        else:
            arguments += ['--mode', 'random']
        await executor.execute_job(
            arguments=arguments,
            **job_block.get_job_variables()
        )

    with h5py.File(Path(parameters.pkgpath) / 'skqd_init.h5', 'r', libver='latest') as source:
        energy_init = source['energy'][()]

    if isinstance(crbm_models, list):
        filename = 'skqd_rcv.h5'
    else:
        filename = 'skqd_rnd.h5'
    with h5py.File(Path(parameters.pkgpath) / filename, 'r', libver='latest') as source:
        energy = source['energy'][()]

    return energy, energy_init


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
