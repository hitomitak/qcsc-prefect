"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import tempfile
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
from skqd_z2lgt.tasks.train_generator import train_generator_flow
from skqd_z2lgt.tasks.diagonalize import check_saved_result


TASK_SCRIPT_DIR = Path(__file__).parents[0] / 'tasks'


@flow
async def skqd_z2lgt(
    parameters: Parameters,
    runtime_name: str = 'ibm-runner',
    cpu_pyfuncjob_name: str = 'cpu-pyfunc',
    cuda_scriptjob_name: str = 'cuda-script'
) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD.

    Args:
        parameters: Configuration parameters.
        runtime_name: Name of QuantumRunner block.
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
        dmrg_future = dmrg.submit(parameters, cpu_pyfuncjob_name=cpu_pyfuncjob_name)
    logger.info('Running a quantum job to obtain the bitstrings')
    sample_quantum_future = sample_quantum.submit(parameters,
                                                  runtime_name=runtime_name)
    logger.info('Correcting and converting link states to plaquette states')
    preprocess_future = preprocess.submit(parameters,
                                          cpu_pyfuncjob_name=cpu_pyfuncjob_name,
                                          wait_for=[sample_quantum_future])
    logger.info('Training conditional restricted Boltzmann machines')
    train_generator_future = train_generator.submit(parameters,
                                                    cuda_scriptjob_name=cuda_scriptjob_name,
                                                    wait_for=[preprocess_future])
    logger.info('Performing SQD with no configuration recovery')
    diagonalize_init_future = diagonalize.submit(parameters, 'init',
                                                 cuda_scriptjob_name=cuda_scriptjob_name,
                                                 wait_for=[preprocess_future])
    logger.info('Performing SQD with random bit flips')
    diagonalize_recov_future = diagonalize.submit(parameters, 'full',
                                                  cuda_scriptjob_name=cuda_scriptjob_name,
                                                  wait_for=[train_generator_future,
                                                            diagonalize_init_future])
    logger.info('Performing SQD with configuration recovery')
    diagonalize_random_future = diagonalize.submit(parameters, 'random',
                                                   cuda_scriptjob_name=cuda_scriptjob_name,
                                                   wait_for=[diagonalize_init_future])
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

    def dmrg_fn(hamiltonian):
        async def fn():
            with tempfile.NamedTemporaryFile(dir=parameters.pkgpath) as tfile:
                filename = tfile.name
            energy = await job_block.run(ising_dmrg, hamiltonian, filename=filename,
                                         nsweeps=dmrg_params.nsweeps,
                                         maxdim=dmrg_params.maxdim, cutoff=dmrg_params.cutoff,
                                         julia_bin=julia_bin)
            states, probs = await job_block.run(get_mps_probs, filename,
                                                num_samples=dmrg_params.num_samples,
                                                julia_bin=julia_bin)
            os.unlink(filename)
            return energy, states, probs

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(fn())).result()

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
    async with asyncio.TaskGroup() as tg:
        runtime_task = tg.create_task(QuantumRuntime.load(runtime_name))
        options_task = tg.create_task(Variable.get(parameters.runtime.options_name))

    runtime = runtime_task.result()
    options = options_task.result()

    def fetch_result_fn():
        def fn():
            job = PrimitiveJobRun(job_id=parameters.runtime.job_id, credentials=runtime.credentials)
            return asyncio.run(job.fetch_result())

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    # Cannot pass runtime.get_target directly as get_target_fn to sample_quantum_flow because of
    # async_dispatch
    def get_target_fn():
        def fn():
            return runtime.get_target()

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    def sample_fn(pubs):
        def fn(pubs):
            return runtime.sampler(sampler_pubs=pubs, options=options)

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn, pubs).result()

    sample_quantum_flow(parameters, fetch_result_fn, get_target_fn, sample_fn, logger)


def convert_bit_arrays(parameters, etype, dual_lattice):
    bit_arrays = load_raw(parameters, etype)
    batch_size = parameters.runtime.shots // 20
    return [convert_link_to_plaq(bit_array, dual_lattice, batch_size) for bit_array in bit_arrays]


@task
async def preprocess(
    parameters: Parameters,
    cpu_pyfuncjob_name: str
):
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

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)

    async def run_train_job(istep):
        with job_block.get_executor() as executor:
            arguments = [TASK_SCRIPT_DIR / 'train_generator.py', parameters.pkgpath, f'{istep}']
            return await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )

    async def run_train_jobs(steps_to_train):
        async with asyncio.TaskGroup() as taskgroup:
            for istep in steps_to_train:
                taskgroup.create_task(run_train_job(istep))

    def train_fn(steps_to_train, _):
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_train_jobs(steps_to_train))).result()

    train_generator_flow(parameters, None, train_fn, False, logger)


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
        group_name = 'skqd_init'
    elif mode == 'full':
        group_name = 'skqd_rcv'
    else:
        group_name = 'skqd_rnd'

    saved_result = check_saved_result(parameters, group_name)
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        return saved_result[1]

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    with job_block.get_executor() as executor:
        arguments = [TASK_SCRIPT_DIR / 'diagonalize.py', parameters.pkgpath, '--mode', mode]
        await executor.execute_job(
            arguments=arguments,
            **job_block.get_job_variables()
        )

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

    with open(args.parameters, 'r', encoding='utf-8') as src:
        params = Parameters(**yaml.load(src, yaml.Loader))

    asyncio.run(skqd_z2lgt(params))
