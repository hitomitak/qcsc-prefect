"""Compose the Krylov circuits and run the quantum runtime sampler."""
import os
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Optional
import h5py
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit.primitives import BitArray, PrimitiveResult
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits


def get_trotter_circuits(
    parameters: Parameters,
    target: Target,
    logger: Optional[logging.Logger] = None
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
    logger = logger or logging.getLogger(__name__)

    if (trotter_steps_per_dt := parameters.circuit.trotter_steps_per_dt) is None:
        trotter_steps_per_dt = [1] * len(parameters.skqd.time_steps)

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    exp_circuits = []
    ref_circuits = []
    for time_step, n_tr in zip(parameters.skqd.time_steps, trotter_steps_per_dt):
        circuits, layout = make_step_circuits(
            lattice, parameters.lgt.plaquette_energy, time_step / n_tr, target,
            charged_vertices=parameters.lgt.charged_vertices,
            layout=parameters.circuit.layout,
            optimization_level=parameters.circuit.optimization_level
        )

        steps = list(range(n_tr, (int(parameters.skqd.num_krylov) + 1) * n_tr, n_tr))
        exp_circuits.append(
            compose_trotter_circuits(circuits[0], circuits[1], circuits[3], circuits[5], steps)
        )
        ref_circuits.append(
            compose_trotter_circuits(circuits[0], circuits[2], circuits[4], circuits[5], steps)
        )
    return layout, exp_circuits, ref_circuits


def save_raw(
    parameters: Parameters,
    pub_result: PrimitiveResult,
    logger: Optional[logging.Logger] = None
):
    """Save the sampled bitstrings to files."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving raw link data')
    dirpath = Path(parameters.pkgpath) / 'data' / 'raw'
    os.makedirs(dirpath, exist_ok=True)

    ires = 0
    for etype in ['exp', 'ref']:
        for idt, time_step in enumerate(parameters.skqd.time_steps):
            for ikr in range(1, parameters.skqd.num_krylov + 1):
                path = dirpath / f'{etype}_dt{idt}_k{ikr}.h5'
                with h5py.File(path, 'w', libver='latest') as out:
                    out.attrs['time_step'] = time_step
                    bit_array = pub_result[ires].data.c
                    dataset = out.create_dataset('link', data=bit_array.array)
                    dataset.attrs['num_bits'] = bit_array.num_bits
                    ires += 1


def load_raw(
    parameters: Parameters,
    etype: Optional[str] = None,
    idt: Optional[int] = None,
    ikrylov: Optional[int] = None
) -> (tuple[list[list[BitArray]], list[list[BitArray]]] | list[list[BitArray]] | list[BitArray]
      | BitArray):
    """Load the sampled bitstrings from files."""
    def read_bit_array(et, itm, ikr):
        filename = Path(parameters.pkgpath) / 'data' / 'raw' / f'{et}_dt{itm}_k{ikr}.h5'
        with h5py.File(filename, 'r', libver='latest') as source:
            dataset = source['link']
            return BitArray(dataset[()], int(dataset.attrs['num_bits']))

    def read_dt_arrays(et, itm):
        return [read_bit_array(et, itm, ikr) for ikr in range(1, parameters.skqd.num_krylov + 1)]

    def read_et_arrays(et):
        return [read_dt_arrays(et, itm) for itm in range(len(parameters.skqd.time_steps))]

    if etype is None:
        return tuple(read_et_arrays(et) for et in ['exp', 'ref'])
    if idt is None:
        return read_et_arrays(etype)
    if ikrylov is None:
        return read_dt_arrays(etype, idt)
    return read_bit_array(etype, idt, ikrylov)


def sample_quantum_flow(
    parameters: Parameters,
    fetch_result_fn: Callable,
    get_target_fn: Callable,
    sample_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """General flow for obtaining bitstring samples from quantum circuits."""
    logger = logger or logging.getLogger(__name__)

    try:
        load_raw(parameters)
    except FileNotFoundError:
        pass
    else:
        logger.info('Raw bitstrings already saved to file')
        return

    if parameters.runtime.job_id:
        logger.info('Fetching result of workload %s', parameters.runtime.job_id)
        pub_result = fetch_result_fn()
        layout = None
    else:
        logger.info('Running a new experiment')
        # Transpile and compose the circuits
        target = get_target_fn()
        layout, exp_circuits, ref_circuits = get_trotter_circuits(parameters, target, logger)
        # Run primitive
        logger.info('Submitting a runtime job')
        pubs = [(circ, [], parameters.runtime.shots_exp) for circ in sum(exp_circuits, [])]
        pubs += [(circ, [], parameters.runtime.shots_ref) for circ in sum(ref_circuits, [])]
        pub_result, job_id = sample_fn(pubs)

        parameters.circuit.layout = layout
        parameters.runtime.job_id = job_id
        with open(Path(parameters.pkgpath) / 'parameters.json', 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())

    save_raw(parameters, pub_result, logger)


def sample_quantum(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
) -> tuple[list[BitArray], list[BitArray]]:
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.


    Returns:
        Lists of BitArrays for forward-evolution and forward-backward circuits.
    """
    logger = logger or logging.getLogger(__name__)
    service = QiskitRuntimeService(instance=parameters.runtime.instance)

    def fetch_result_fn():
        return service.job(parameters.runtime.job_id).result()

    def get_target_fn():
        backend = service.backend(parameters.runtime.backend, use_fractional_gates=True)
        return backend.target

    def sample_fn(pubs):
        backend = service.backend(parameters.runtime.backend, use_fractional_gates=True)
        options = dict(parameters.runtime.options)
        sampler = Sampler(backend, options=options)
        job = sampler.run(pubs)
        logger.info('Sampler job: %s', job.job_id())
        return job.result(), job.job_id()

    return sample_quantum_flow(parameters, fetch_result_fn, get_target_fn, sample_fn, logger)
