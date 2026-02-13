from __future__ import annotations

import argparse
import asyncio

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

try:
    from examples.miyabi_prefect_bitcount_demo.wrapper_block import BITLEN, BitCounterWrapperBlock
except ModuleNotFoundError:
    # Supports direct script execution:
    # python examples/miyabi_prefect_bitcount_demo/flow_wrapper.py ...
    from wrapper_block import BITLEN, BitCounterWrapperBlock


@flow(name="miyabi-bitcount-wrapper-flow")
async def miyabi_bitcount_wrapper_flow(
    *,
    runtime_block_name: str = "ibm-runner",
    counter_block_name: str = "bit-counter-wrapper-demo",
    options_variable_name: str = "miyabi-bitcount-options",
    default_shots: int = 100000,
):
    runtime = await QuantumRuntime.load(runtime_block_name)
    counter = await BitCounterWrapperBlock.load(counter_block_name)

    options = await Variable.get(
        options_variable_name,
        default={"params": {"shots": default_shots}},
    )

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

    results = await runtime.sampler([(isa,)], options=options)
    bitstrings = results[0].data.meas.get_bitstrings()

    counts = await counter.get(bitstrings)

    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict-wrapper",
    )

    return {
        "mode": "wrapper",
        "shots": int(sum(counts.values())),
        "num_unique_bitstrings": len(counts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wrapper-style Miyabi BitCount tutorial flow.")
    parser.add_argument("--runtime-block", default="ibm-runner")
    parser.add_argument("--counter-block", default="bit-counter-wrapper-demo")
    parser.add_argument("--options-variable", default="miyabi-bitcount-options")
    parser.add_argument("--shots", type=int, default=100000)
    args = parser.parse_args()

    print(
        asyncio.run(
            miyabi_bitcount_wrapper_flow(
                runtime_block_name=args.runtime_block,
                counter_block_name=args.counter_block,
                options_variable_name=args.options_variable,
                default_shots=args.shots,
            )
        )
    )
