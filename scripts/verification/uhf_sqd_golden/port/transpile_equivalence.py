"""PR3: the source's SABRE compilation vs the port's QRMI hook, on a fake 127-qubit backend.

Usage: python -I transpile_equivalence.py SOURCE_TRANSPILE_CUSTOM_PY FCIDUMP
For walker 0 and walker 1 LUCJ circuits of the NH2 reference:
  source: find_optimal_layout(circuit) then transpile_circuit(circuit, layout)  (3968e58)
  port:   prepare(circuit) then transpile(prepared, target, config)
and, as the source did within one flow run, walker 1 compiled with walker 0's cached layout.
"""

import importlib.util
import sys
import time
from types import SimpleNamespace

src_path, fcidump = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("source_transpile_custom", src_path)
src = importlib.util.module_from_spec(spec)
spec.loader.exec_module(src)
src.create_table_artifact = lambda **kwargs: None
src.get_run_logger = lambda: SimpleNamespace(
    info=lambda *a, **k: None, warning=lambda *a, **k: None
)

from qcsc_workflow_utility import uhf  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke  # noqa: E402
from uhf_sqd import lucj, transpilation  # noqa: E402
from uhf_sqd.flow_params import CircuitParameters  # noqa: E402
from uhf_sqd.main import _build_ab_indices  # noqa: E402

ep = uhf.compute_uhf_integrals_from_fcidump.fn(fcidump, cache_dir=None)
norb = ep.num_orbitals
aa = [(p, p + 1) for p in range(norb - 1)]
ab = _build_ab_indices(norb, 4)
params = lucj.initialize_ucj_parameters.fn(
    ep, aa, ab, num_walkers=2, randomization_factor=0.2, n_lucj_layers=2
)
circuits = [lucj.create_lucj_circuit.fn(p, ep, aa, ab, 2, True) for p in params]
target = FakeSherbrooke().target
cp = CircuitParameters(sabre_layout_trials=64)  # same settings on both sides
config = transpilation.transpile_config(cp)


def source_compile(circuit, layout=None):
    if layout is None:
        layout = src.find_optimal_layout.fn(
            test_circuit=circuit,
            target=target,
            optimization_level=cp.optimization_level,
            max_iterations=cp.sabre_max_iterations,
            swap_trials=cp.sabre_swap_trials,
            layout_trials=cp.sabre_layout_trials,
        )
    return src.transpile_circuit.fn(
        circuit=circuit, target=target, layout=layout, optimization_level=cp.optimization_level
    ), layout


def phys(isa):
    return sorted(isa.layout.initial_virtual_layout(filter_ancillas=True).get_physical_bits())


results = {}
for w, circuit in enumerate(circuits):
    t0 = time.perf_counter()
    isa_src, layout = source_compile(circuit)
    t1 = time.perf_counter()
    isa_port = transpilation.transpile(transpilation.prepare(circuit).circuit, target, config)
    t2 = time.perf_counter()
    results[w] = (isa_src, layout)
    same = isa_src == isa_port
    print(
        f"walker {w}: source == port ISA circuit: {same}; same physical qubits: "
        f"{phys(isa_src) == phys(isa_port)}; 2q gates src/port: "
        f"{isa_src.count_ops().get('ecr', 0)}/{isa_port.count_ops().get('ecr', 0)}; "
        f"compile s src/port: {t1 - t0:.1f}/{t2 - t1:.1f}"
    )
# The source's RUN_ID cache: walker 1 compiled with walker 0's layout.
cached, _ = source_compile(circuits[1], layout=results[0][1])
own = results[1][0]
print(
    f"walker 1 with walker 0's cached layout == walker 1 with its own layout: {cached == own}; "
    f"same physical qubits: {phys(cached) == phys(own)}; 2q gates cached/own: "
    f"{cached.count_ops().get('ecr', 0)}/{own.count_ops().get('ecr', 0)}"
)
