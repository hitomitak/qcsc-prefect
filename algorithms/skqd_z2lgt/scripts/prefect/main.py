"""Definition of the main Prefect flow for Z2LGT SKQD."""

import os
from pathlib import Path
import logging
import asyncio
import tempfile
from typing import Optional
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

TASK_SCRIPT_DIR = Path(__file__).parents[1] / 'tasks'


class Configuration(BaseModel):
    """Workflow configuration parameters."""

    lattice: str = Field(
        description='Lattice configuration.',
        title='Lattice configuration'
    )
    plaquette_energy: float = Field(
        description='Plaquette energy in the Hamiltonian (transverse field strength in the dual'
                    ' Ising model).',
        title='Plaquette energy'
    )
    delta_t: float = Field(
        description='Time step size in units of inverse electric energy.',
        title='Time step'
    )
    num_steps: int = Field(
        description='Number of Trotter time steps to simulate.',
        title='Number of time steps'
    )
    shots: int = Field(
        default=100_000,
        description='Number of shots per circuit.',
        title='Number of shots'
    )
    basis_2q: str = Field(
        default='rzz',
        description='Two-qubit gate used in the Trotter circuit. Note that the selection "rzz" will'
                    ' utilize CZ gates together with Rzz.',
        title='Base two-qubit gate'
    )


@flow
async def main(
    configuration: Configuration,
    runner_name: str = 'ibm-runner',
    option_name: str = 'sampler_options',
    cpu_pyfuncjob_name: str = 'pyfunc-qii-miyabi-kawasaki',
    cuda_scriptjob_name: str = 'script-qii-miyabi-kawasaki',
    runtime_job_id: Optional[str] = None,
    output_filename: Optional[str] = None
) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD.

    Args:
        configuration: SKQD configuration.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable for QuantumRunner sampler options.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        cuda_scriptjob_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
        output_filename: If specified, the name of the HDF5 file where intermediate and final output
            of the workflow are written. If left unspecified, the file is considered temporary and
            will be deleted at the end of the workflow.
    """
    logger = get_run_logger()
    logger.setLevel(logging.INFO)

    output_filename, cleanup_output = open_output(configuration, output_filename)
    logger.info('Running a quantum job to obtain the bitstrings')
    bit_arrays = await sample_krylov_bitstrings(configuration, runner_name, option_name,
                                                runtime_job_id, output_filename)
    logger.info('Correcting and converting link states to plaquette states')
    await preprocess_bitstrings(configuration, cpu_pyfuncjob_name, bit_arrays, output_filename)
    logger.info('Training conditional restricted Boltzmann machines')
    await train_crbm(cuda_scriptjob_name, output_filename)
    logger.info('Performing SQD with configuration recovery')
    await project_and_diagonalize(cuda_scriptjob_name, output_filename)

    with h5py.File(output_filename) as source:
        energy = source['skqd_rcv/energy'][()]
        eigvec = source['skqd_rcv/eigvec'][()]

    logger.info('Estimated ground-state energy is %f', energy)

    if cleanup_output:
        os.unlink(output_filename)

    return energy, eigvec


@task
def open_output(
    configuration: Configuration,
    output_filename: Optional[str] = None
) -> tuple[str, bool]:
    """Open a new output HDF5 file and set it up for the workflow, or validate an existing file.

    Args:
        configuration: SKQD configuration.
        output_filename: If specified, the name of the HDF5 file where intermediate and final output
            of the workflow are written. If left unspecified, the file is considered temporary and
            will be deleted at the end of the workflow.

    Returns:
        The name of the output file and the boolean flag indicating the file as temporary.
    """
    cleanup_output = False
    if output_filename is None:
        cleanup_output = True
        with tempfile.NamedTemporaryFile() as tfile:
            output_filename = tfile.name

    try:
        with h5py.File(output_filename, 'r') as source:
            for key, value in configuration.model_dump().items():
                if ((isinstance(value, float) and not np.isclose(source.attrs[key], value))
                        or (not isinstance(value, float) and source.attrs[key] != value)):
                    raise RuntimeError(f'Configuration {key} does not match what is saved in'
                                       f' existing file {output_filename}:'
                                       f' {source.attrs[key]} != {value}')
    except FileNotFoundError:
        with h5py.File(output_filename, 'w-', libver='latest') as output_file:
            for key, value in configuration.model_dump().items():
                output_file.attrs[key] = value

    return output_filename, cleanup_output


@task
def get_trotter_circuits(
    configuration: Configuration,
    target: Target,
    optimization_level: int = 2
) -> tuple[list[int], list[QuantumCircuit], list[QuantumCircuit]]:
    """Compose full Trotter simulation circuits.

    We first generate single-step circuit elements for the given lattice and base two-qubit gate,
    compile them, then compose the resulting ISA circuits into multi-step Trotter simulation
    circuits. Both the forward-evolution (Krylov) circuits and forward-backward (reference) circuits
    are returned.

    Args:
        configuration: SKQD configuration.
        target: Backend target.
        opimization_level: Transpiler optimization level.

    Returns:
        Physical qubit layout and lists (length configuration.num_steps) of forward-evolution and
        forward-backward circuits.
    """
    lattice = TriangularZ2Lattice(configuration.lattice)
    layout = lattice.layout_heavy_hex(target=target, basis_2q=configuration.basis_2q)

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
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
    circuits = make_step_circuits(lattice, configuration.plaquette_energy,
                                  configuration.delta_t, configuration.basis_2q)
    # Somehow the combination of multiprocessing pm.run + prefect task causes the former to hang
    full_step, fwd_step, bkd_step, measure = [pm.run(circuit) for circuit in circuits]

    id_step = fwd_step.compose(bkd_step)
    exp_circuits = compose_trotter_circuits(full_step, measure, configuration.num_steps)
    ref_circuits = compose_trotter_circuits(id_step, measure, configuration.num_steps)
    return layout, exp_circuits, ref_circuits


@task
async def sample_krylov_bitstrings(
    configuration: Configuration,
    runner_name: str,
    option_name: str,
    runtime_job_id: str | None,
    output_filename: str
) -> tuple[list[BitArray], list[BitArray]]:
    """Run the circuits on a backend and return the sampler results.

    Args:
        configuration: SKQD configuration.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable storing sampler primitive options.
        output_filename: Name of the HDF5 file where intermediate and final output of the workflow
            are written.

    Returns:
        Lists of BitArrays for forward-evolution and forward-backward circuits.
    """
    with h5py.File(output_filename, 'r') as source:
        try:
            group = source['data/raw']
        except KeyError:
            pass
        else:
            num_steps = source.attrs['num_steps']
            dlists = ([], [])
            for etype, dlist in zip(['exp', 'ref'], dlists):
                for istep in range(num_steps):
                    dataset = group[f'{etype}_step{istep}']
                    dlist.append(BitArray(dataset[()], int(dataset.attrs['num_bits'])))

            return dlists

    runtime = await QuantumRuntime.load(runner_name)

    if runtime_job_id:
        job = PrimitiveJobRun(job_id=runtime_job_id, credentials=runtime.credentials)
        pub_result = await job.fetch_result()
        layout = None
    else:
        options = await Variable.get(option_name)

        # Transpile and compose the circuits
        target = await runtime.get_target()
        layout, exp_circuits, ref_circuits = get_trotter_circuits(configuration, target)

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
            group.attrs['layout'] = layout
        for ires, res in enumerate(pub_result):
            if ires < configuration.num_steps:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = ires % configuration.num_steps
            bit_array = res.data.c
            dataset = group.create_dataset(f'{etype}_step{istep}', data=bit_array.array)
            dataset.attrs['num_bits'] = bit_array.num_bits

    return ([res.data.c for res in pub_result[:configuration.num_steps]],
            [res.data.c for res in pub_result[configuration.num_steps:]])


@task
async def preprocess_bitstrings(
    configuration: Configuration,
    cpu_pyfuncjob_name: str,
    bit_arrays: tuple[list[BitArray], list[BitArray]],
    output_filename: str
):
    logger = get_run_logger()

    with h5py.File(output_filename, 'r') as source:
        data_group = source.get('data', {})
        if 'vtx' in data_group and 'plaq' in data_group:
            logger.info('vtx and plaq data already exist.')
            return

    job_block = await PyFunctionJob.load(cpu_pyfuncjob_name)
    lattice = TriangularZ2Lattice(configuration.lattice)
    dual_lattice = lattice.plaquette_dual()
    batch_size = configuration.shots // 20

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
            if idx < configuration.num_steps:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = idx % configuration.num_steps
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
    cuda_scriptjob_name: str,
    output_filename: str
):
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

    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    with job_block.get_executor() as executor:
        tasks = []
        async with asyncio.TaskGroup() as taskgroup:
            for istep in steps:
                with tempfile.NamedTemporaryFile(dir=executor.work_dir) as tfile:
                    pass

                arguments = [
                    TASK_SCRIPT_DIR / 'train_crbm.py',
                    output_filename,
                    f'{istep}',
                    '--out-filename', tfile.name,
                    '--num-epochs', '100',
                    '--rtol', '2.'
                ]
                atask = taskgroup.create_task(
                    executor.execute_job(
                        arguments=arguments,
                        **job_block.get_job_variables()
                    )
                )
                tasks.append((istep, atask, tfile.name))

                logger.info('Step %d will write trained model to %s', istep, tfile.name)

        with h5py.File(output_filename, 'r+') as out:
            for istep, atask, tempname in tasks:
                if (code := atask.result()) != 0:
                    raise RuntimeError(f'CRBM training return code {code} for Trotter step {istep}')
                group = out['crbm']
                try:
                    del group[f'step{istep}']
                except KeyError:
                    pass

                with h5py.File(tempname, 'r') as source:
                    source.copy(f'crbm/step{istep}', group)


@task
async def project_and_diagonalize(
    cuda_scriptjob_name: str,
    output_filename: str
):
    job_block = await MiyabiJobBlock.load(cuda_scriptjob_name)
    with job_block.get_executor() as executor:
        arguments = [
            TASK_SCRIPT_DIR / 'skqd_recovery.py',
            output_filename,
            '--gpu', 'all',
            '--num-gen', '3',
            '--niter', '10',
            '--terminate', 'diff=0.01', 'dim=1000000'
        ]
        await executor.execute_job(
            arguments=arguments,
            **job_block.get_job_variables()
        )
