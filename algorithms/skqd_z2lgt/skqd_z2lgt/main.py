"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import h5py
from prefect import flow, task, get_run_logger
from prefect.client.schemas.filters import (ArtifactFilter, ArtifactFilterKey,
                                            ArtifactFilterTaskRunId)
from prefect.client.orchestration import get_client
from prefect.variables import Variable
from prefect.runtime import task_run
from prefect_qiskit.runtime import QuantumRuntime
from prefect_qiskit.primitives import PrimitiveJobRun
from prefect_miyabi import MiyabiJobBlock, PyFunctionJob
from skqd_z2lgt.ising_dmrg import ising_dmrg, get_mps_probs
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.open_output import open_output
from skqd_z2lgt.tasks.dmrg import dmrg_flow
from skqd_z2lgt.tasks.sample_quantum import sample_quantum_flow
from skqd_z2lgt.tasks.preprocess import preprocess_flow
from skqd_z2lgt.tasks.train_generator import train_generator_flow
from skqd_z2lgt.tasks.diagonalize import check_saved_result
from skqd_z2lgt.tasks.common import make_dual_lattice

TASK_SCRIPT_DIR = Path(__file__).parents[0] / 'tasks'


@flow
async def skqd_z2lgt(
    parameters: Parameters,
    runtime_name: str = 'ibm-runner',
    cpu_pyfuncjob_name: str = 'cpu-pyfunc',
    cpu_scriptjob_name: str = 'cpu-script',
    cuda_scriptjob_name: str = 'cuda-script'
) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD.

    Args:
        parameters: Configuration parameters.
        runtime_name: Name of QuantumRunner block.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        cpu_scriptjob_name: Name of the MiyabiJobBlock that executes a python script.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes a python script in a CUDA
            environment.
    """
    logger = get_run_logger()
    logger.setLevel(logging.INFO)

    tmpdir = None
    if not parameters.pkgpath:
        tmpdir = tempfile.TemporaryDirectory()
        parameters.pkgpath = tmpdir.name

    open_output(parameters, logger)
    if parameters.dmrg:
        dmrg_future = dmrg.submit(parameters, cpu_pyfuncjob_name=cpu_pyfuncjob_name)
    sample_quantum_future = sample_quantum.submit(parameters,
                                                  runtime_name=runtime_name)
    preprocess_future = preprocess.submit(parameters,
                                          cpu_scriptjob_name=cpu_scriptjob_name,
                                          wait_for=[sample_quantum_future])
    train_generator_future = train_generator.submit(parameters,
                                                    cuda_scriptjob_name=cuda_scriptjob_name,
                                                    wait_for=[preprocess_future])
    diagonalize_init_future = diagonalize.submit(parameters, 'init',
                                                 cuda_scriptjob_name=cuda_scriptjob_name,
                                                 wait_for=[preprocess_future])
    diagonalize_random_future = diagonalize.submit(parameters, 'rnd',
                                                   cuda_scriptjob_name=cuda_scriptjob_name,
                                                   wait_for=[diagonalize_init_future])
    diagonalize_recov_future = diagonalize.submit(parameters, 'rcv',
                                                  cuda_scriptjob_name=cuda_scriptjob_name,
                                                  wait_for=[train_generator_future,
                                                            diagonalize_init_future])
    energy_norecov = diagonalize_init_future.result()
    energy_random = diagonalize_random_future.result()
    energy = diagonalize_recov_future.result()

    if tmpdir:
        tmpdir.cleanup()

    if parameters.dmrg:
        dmrg_energy = dmrg_future.result()
        logger.info('DMRG energy: %f', dmrg_energy)
    logger.info('SKQD energy (no conf. recovery): %f', energy_norecov)
    logger.info('SKQD energy (random bit flips): %f', energy_random)
    logger.info('SKQD energy (full conf. recovery): %f', energy)

    return energy


@task
async def dmrg(
    parameters: Parameters,
    cpu_pyfuncjob_name: str
) -> float:
    """Run DMRG and MPS sampling."""
    logger = get_run_logger()
    logger.info('Estimating ground-state energy via DMRG')

    def run_dmrg_and_sampling(_parameters):
        with tempfile.NamedTemporaryFile() as tfile:
            filename = tfile.name

        dual_lattice = make_dual_lattice(_parameters)
        hamiltonian = dual_lattice.make_hamiltonian(_parameters.lgt.plaquette_energy)
        dp = _parameters.dmrg
        julia_bin = 'julia'
        if dp.julia_sysimage:
            julia_bin = ['julia', '--sysimage', dp.julia_sysimage]

        energy = ising_dmrg(hamiltonian, filename=filename, nsweeps=dp.nsweeps, maxdim=dp.maxdim,
                            cutoff=dp.cutoff, julia_bin=julia_bin)
        states, probs = get_mps_probs(filename, num_samples=dp.num_samples,
                                      num_threads=os.cpu_count(), julia_bin=julia_bin)
        os.unlink(filename)
        return energy, states, probs

    job_block = await PyFunctionJob.load(cpu_pyfuncjob_name)

    def dmrg_fn():
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(
                lambda: asyncio.run(job_block.run(run_dmrg_and_sampling, parameters))
            ).result()

    return dmrg_flow(parameters, dmrg_fn, logger)


@task
async def sample_quantum(
    parameters: Parameters,
    runtime_name: str = 'ibm-runner'
):
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable storing sampler primitive options.
    """
    logger = get_run_logger()
    logger.info('Sampling Trotter circuit final state bitstrings')

    async with asyncio.TaskGroup() as tg:
        runtime_task = tg.create_task(QuantumRuntime.load(runtime_name))
        options_task = tg.create_task(Variable.get(parameters.runtime.options_name))

    runtime = runtime_task.result()
    options = options_task.result()
    task_id = task_run.id

    def fetch_result_fn():
        def fn():
            job = PrimitiveJobRun(job_id=parameters.runtime.job_id, credentials=runtime.credentials)
            return asyncio.run(job.fetch_result())

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    # Cannot pass runtime.get_target directly as get_target_fn to sample_quantum_flow because of
    # async_dispatch (get_target would be called in a thread running an event loop and will
    # therefore return a coroutine)
    def get_target_fn():
        def fn():
            return runtime.get_target()

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    def sample_fn(pubs):
        def fn(pubs):
            return runtime.sampler(sampler_pubs=pubs, options=options)

        with ThreadPoolExecutor(1) as executor:
            pub_result = executor.submit(fn, pubs).result()

        prefect_client = get_client(sync_client=True)
        artifacts = prefect_client.read_artifacts(
            limit=1,
            artifact_filter=ArtifactFilter(
                key=ArtifactFilterKey(any_=['job-metrics']),
                task_run_id=ArtifactFilterTaskRunId(any_=[task_id])
            )
        )
        if artifacts:
            keys, values = json.loads(artifacts[0].data)
            job_id = values[keys.index('job_id')]
        else:
            job_id = ''

        return pub_result, job_id

    sample_quantum_flow(parameters, fetch_result_fn, get_target_fn, sample_fn, logger)


@task
async def preprocess(
    parameters: Parameters,
    cpu_scriptjob_name: str
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpu_scriptjob_name: Name of the MiyabiJobBlock that executes a python script.
    """
    logger = get_run_logger()
    logger.info('Correcting and converting link states to plaquette states')

    job_block = await MiyabiJobBlock.load(cpu_scriptjob_name)
    job_block.num_nodes = 2 * parameters.skqd.n_trotter_steps
    job_block.mpiprocs = 1
    job_block.walltime = '00:03:00'

    async def convert_miyabi():
        with job_block.get_executor() as executor:
            arguments = [TASK_SCRIPT_DIR / 'preprocess.py', parameters.pkgpath, '--mpi']
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError('PBS job preprocess.py failed')

    def convert_fn():
        with ThreadPoolExecutor(1) as executor:
            executor.submit(lambda: asyncio.run(convert_miyabi())).result()

    preprocess_flow(parameters, convert_fn, logger=logger)


@task
async def train_generator(
    parameters: Parameters,
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
    logger.info('Training conditional restricted Boltzmann machines')

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    job_block.mpiprocs = 1
    job_block.walltime = '01:00:00'

    async def run_train_job(steps_to_train):
        job_block.num_nodes = len(steps_to_train)
        with job_block.get_executor() as executor:
            arguments = [TASK_SCRIPT_DIR / 'train_generator.py', parameters.pkgpath, '--mpi']
            arguments += ['--istep'] + [f'{istep}' for istep in steps_to_train]
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError('PBS job train_generator.py failed')

    def train_fn(steps_to_train):
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_train_job(steps_to_train))).result()

    train_generator_flow(parameters, train_fn, logger=logger)


@task
async def diagonalize(
    parameters: Parameters,
    mode: str,
    cuda_scriptjob_name: str
) -> float:
    """Perform SQD with iterative configuration recovery.

    Args:
        parameters: Configuration parameters.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
    """
    logger = get_run_logger()
    if mode == 'init':
        logger.info('Performing SQD with no configuration recovery')
    elif mode == 'rnd':
        logger.info('Performing SQD with random bit flips')
    else:
        logger.info('Performing SQD with configuration recovery')

    group_name = f'skqd_{mode}'
    saved_result = check_saved_result(parameters, group_name)
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        return saved_result[1]

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    job_block.launcher = 'single'
    job_block.num_nodes = 1
    job_block.walltime = '02:00:00'
    with job_block.get_executor() as executor:
        arguments = [TASK_SCRIPT_DIR / 'diagonalize.py', parameters.pkgpath, '--mode', mode]
        exit_status = await executor.execute_job(
            arguments=arguments,
            **job_block.get_job_variables()
        )
    if exit_status != 0:
        raise RuntimeError(f'PBS job "diagonalize.py --mode {mode}" failed')

    with h5py.File(Path(parameters.pkgpath) / f'{group_name}.h5', 'r', libver='latest') as source:
        return source['energy'][()]


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


if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser(prog='skqd_z2lgt')
    parser.add_argument('parameters', metavar='PATH',
                        help='Path to a yaml file containing the workflow parameters.')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')

    if os.path.isdir(args.parameters):
        with open(Path(args.parameters) / 'parameters.json', 'r', encoding='utf-8') as source:
            params = Parameters.model_validate_json(source.read())
    else:
        with open(args.parameters, 'r', encoding='utf-8') as src:
            params = Parameters(**yaml.load(src, yaml.Loader))

    asyncio.run(skqd_z2lgt(params))
