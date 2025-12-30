"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
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
    diagonalize_future = diagonalize.submit(parameters, cuda_scriptjob_name=cuda_scriptjob_name,
                                            wait_for=[train_generator_future])
    energy_cr, energy_rn = diagonalize_future.result()

    if tmpdir:
        tmpdir.cleanup()

    if parameters.dmrg:
        dmrg_energy = dmrg_future.result()
        logger.info('DMRG energy: %f', dmrg_energy)
    logger.info('SKQD energy (random bit flips): %f', energy_rn)
    logger.info('SKQD energy (full conf. recovery): %f', energy_cr)

    return energy_cr


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

    async def convert_miyabi(task_specs):
        job_block = await MiyabiJobBlock.load(cpu_scriptjob_name)
        job_block.num_nodes = len(task_specs)
        job_block.mpiprocs = 1
        job_block.walltime = '00:10:00'

        with job_block.get_executor() as executor:
            arguments = [
                TASK_SCRIPT_DIR / 'preprocess.py',
                parameters.pkgpath,
                '--mpi',
                '--etype', ','.join(task[0] for task in task_specs),
                '--idt', ','.join(str(task[1]) for task in task_specs),
                '--ikrylov', ','.join(str(task[2]) for task in task_specs)
            ]
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError('PBS job preprocess.py failed')

    def convert_fn(task_specs):
        with ThreadPoolExecutor(1) as executor:
            executor.submit(lambda: asyncio.run(convert_miyabi(task_specs))).result()

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

    async def run_train_job(task_specs):
        job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
        job_block.mpiprocs = 1
        job_block.num_nodes = len(task_specs)
        job_block.walltime = '01:00:00'

        with job_block.get_executor() as executor:
            arguments = [
                TASK_SCRIPT_DIR / 'train_generator.py',
                parameters.pkgpath,
                '--mpi',
                '--idt', ','.join(str(task[0]) for task in task_specs),
                '--ikrylov', ','.join(str(task[1]) for task in task_specs)
            ]
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError('PBS job train_generator.py failed')

    def train_fn(task_specs):
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_train_job(task_specs))).result()

    train_generator_flow(parameters, train_fn, logger=logger)


@task
async def diagonalize(
    parameters: Parameters,
    cuda_scriptjob_name: str
) -> float:
    """Perform SQD with iterative configuration recovery.

    Args:
        parameters: Configuration parameters.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
    """
    logger = get_run_logger()

    gen_modes = []
    energies = []
    for gen_mode in ['cr', 'rn']:
        if (res := check_saved_result(parameters, f'skqd_{gen_mode}')) is None:
            gen_modes.append(gen_mode)
        else:
            energies.append(res[0])

    if not gen_modes:
        logger.info('All SQD results found on disk.')
        return tuple(energies)

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    job_block.mpiprocs = 1
    job_block.num_nodes = len(gen_modes)
    job_block.walltime = '02:00:00'
    with job_block.get_executor() as executor:
        arguments = [
            TASK_SCRIPT_DIR / 'diagonalize.py',
            parameters.pkgpath,
            '--mpi',
            '--mode', ','.join(gen_modes)
        ]
        exit_status = await executor.execute_job(
            arguments=arguments,
            **job_block.get_job_variables()
        )
    if exit_status != 0:
        raise RuntimeError('PBS job diagonalize.py failed')

    energies = []
    for gen_mode in ['cr', 'rn']:
        res = check_saved_result(parameters, f'skqd_{gen_mode}')
        energies.append(res[1])
    return tuple(energies)


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
        with open(Path(args.parameters) / 'parameters.json', 'r', encoding='utf-8') as src:
            params = Parameters.model_validate_json(src.read())
    else:
        with open(args.parameters, 'r', encoding='utf-8') as src:
            params = Parameters(**yaml.load(src, yaml.Loader))

    asyncio.run(skqd_z2lgt(params))
