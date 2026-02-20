from __future__ import annotations

import asyncio

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

try:
    from get_counts_integration import BITLEN, BitCounter
except ModuleNotFoundError:
    from examples.prefect_bitcount_demo.get_counts_integration import BITLEN, BitCounter


@flow(name="miyabi_tutorial")
async def main():
    runtime = await QuantumRuntime.load("ibm-runner")
    counter = await BitCounter.load("miyabi-tutorial")
    options = await Variable.get("miyabi-tutorial")

    target = await runtime.get_target()
    qc_ghz = QuantumCircuit(BITLEN)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, BITLEN))
    qc_ghz.measure_active()

    pm = generate_preset_pass_manager(
        optimization_level=3,
        target=target,
        seed_transpiler=123,
    )
    isa = pm.run(qc_ghz)
    pub_like = (isa,)

    results = await runtime.sampler([pub_like], options=options)
    bitstrings = results[0].data.meas.get_bitstrings()
    counts = await counter.get(bitstrings)

    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict",
    )


if __name__ == "__main__":
    asyncio.run(main())
