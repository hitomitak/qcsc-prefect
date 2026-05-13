from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from prefect import flow, get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QISKIT_PACKAGE_SRC = PROJECT_ROOT / "packages/qcsc-prefect-qiskit/src"
if str(QISKIT_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(QISKIT_PACKAGE_SRC))

_qiskit_integration = import_module("qcsc_prefect.integrations.qiskit")
QiskitRuntimeConfig = _qiskit_integration.QiskitRuntimeConfig
build_qiskit_estimator_input_digest = _qiskit_integration.build_qiskit_estimator_input_digest
build_qiskit_sampler_input_digest = _qiskit_integration.build_qiskit_sampler_input_digest
fetch_qiskit_job_result_task = _qiskit_integration.fetch_qiskit_job_result_task
qiskit_estimator_submit_cache_key = _qiskit_integration.qiskit_estimator_submit_cache_key
qiskit_result_fetch_cache_key = _qiskit_integration.qiskit_result_fetch_cache_key
qiskit_retry_delays = _qiskit_integration.qiskit_retry_delays
qiskit_sampler_submit_cache_key = _qiskit_integration.qiskit_sampler_submit_cache_key
run_estimator_task = _qiskit_integration.run_estimator_task
run_sampler_task = _qiskit_integration.run_sampler_task
should_retry_qiskit_fetch_failure = _qiskit_integration.should_retry_qiskit_fetch_failure
submit_estimator_job_task = _qiskit_integration.submit_estimator_job_task
submit_sampler_job_task = _qiskit_integration.submit_sampler_job_task

PrimitiveSelection = Literal["all", "sampler", "estimator"]
ModeSelection = Literal["all", "simple", "robust"]
CacheScopeSelection = Literal["flow", "global"]


def _build_sampler_circuit() -> Any:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    circuit.metadata = {"demo": "native_qiskit_sampler_bell"}
    return circuit


def _build_estimator_circuit() -> Any:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.metadata = {"demo": "native_qiskit_estimator_bell"}
    return circuit


def _build_estimator_observable() -> Any:
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_list([("ZZ", 1.0)])


async def _load_runtime_config(runtime_block_name: str) -> Any:
    loaded = QiskitRuntimeConfig.load(runtime_block_name)
    if inspect.isawaitable(loaded):
        return await loaded
    return loaded


async def _prepare_pubs(runtime_block_name: str) -> tuple[list[Any], list[Any], str]:
    from qiskit.transpiler import generate_preset_pass_manager

    runtime_config = await _load_runtime_config(runtime_block_name)
    backend = runtime_config.get_backend()
    backend_name = _backend_name(backend) or runtime_config.backend_name

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=123,
    )
    sampler_circuit = pass_manager.run(_build_sampler_circuit())
    estimator_circuit = pass_manager.run(_build_estimator_circuit())
    observable = _build_estimator_observable()
    if getattr(estimator_circuit, "layout", None) is not None:
        observable = observable.apply_layout(estimator_circuit.layout)

    return [sampler_circuit], [(estimator_circuit, observable)], backend_name


def _backend_name(backend: Any) -> str | None:
    for attr in ("name", "backend_name"):
        value = getattr(backend, attr, None)
        if callable(value):
            value = value()
        if value is not None:
            return str(value)
    return None


def _job_summary(task_output: dict[str, Any]) -> dict[str, Any]:
    metadata = task_output.get("metadata")
    result = task_output.get("result")
    result_len: int | None = None
    try:
        result_len = len(result)
    except TypeError:
        result_len = None

    return {
        "primitive": task_output.get("primitive"),
        "backend_name": task_output.get("backend_name"),
        "job_id": task_output.get("job_id"),
        "shots": task_output.get("shots"),
        "precision": task_output.get("precision"),
        "input_digest": task_output.get("input_digest"),
        "metadata_job_id": getattr(metadata, "job_id", None),
        "result_type": type(result).__name__ if result is not None else None,
        "result_len": result_len,
    }


def _reference_summary(job_reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "primitive": job_reference.get("primitive"),
        "backend_name": job_reference.get("backend_name"),
        "job_id": job_reference.get("job_id"),
        "shots": job_reference.get("shots"),
        "precision": job_reference.get("precision"),
        "input_digest": job_reference.get("input_digest"),
    }


def _sampler_submit_task(*, enable_cache: bool, cache_expiration_days: int) -> Any:
    if not enable_cache:
        return submit_sampler_job_task
    return submit_sampler_job_task.with_options(
        cache_key_fn=qiskit_sampler_submit_cache_key,
        cache_expiration=timedelta(days=cache_expiration_days),
        persist_result=True,
    )


def _estimator_submit_task(*, enable_cache: bool, cache_expiration_days: int) -> Any:
    if not enable_cache:
        return submit_estimator_job_task
    return submit_estimator_job_task.with_options(
        cache_key_fn=qiskit_estimator_submit_cache_key,
        cache_expiration=timedelta(days=cache_expiration_days),
        persist_result=True,
    )


def _fetch_task(*, enable_retry: bool, enable_result_cache: bool) -> Any:
    options: dict[str, Any] = {}
    if enable_result_cache:
        options.update(
            {
                "cache_key_fn": qiskit_result_fetch_cache_key,
                "persist_result": True,
                "result_serializer": "compressed/pickle",
            }
        )
    if enable_retry:
        options.update(
            {
                "retries": len(qiskit_retry_delays()),
                "retry_delay_seconds": qiskit_retry_delays(),
                "retry_condition_fn": should_retry_qiskit_fetch_failure,
            }
        )
    if not options:
        return fetch_qiskit_job_result_task
    return fetch_qiskit_job_result_task.with_options(**options)


def _save_runtime_block(
    *,
    block_name: str,
    backend_name: str,
    channel: str | None,
    instance: str | None,
    token_env: str | None,
    account_name: str | None,
    filename: str | None,
) -> None:
    register = getattr(QiskitRuntimeConfig, "register_type_and_schema", None)
    if callable(register):
        register()

    token = os.getenv(token_env, "").strip() if token_env else None
    QiskitRuntimeConfig(
        backend_name=backend_name,
        channel=channel,
        instance=instance,
        token=token or None,
        account_name=account_name,
        filename=filename,
    ).save(block_name, overwrite=True)


def _selected(value: str, target: str) -> bool:
    return value == "all" or value == target


@flow(name="native-qiskit-primitives-live-test")
async def native_qiskit_primitives_live_test_flow(
    runtime_block_name: str,
    primitive: PrimitiveSelection = "all",
    mode: ModeSelection = "all",
    shots: int = 100,
    precision: float = 0.2,
    artifact_prefix: str = "native-qiskit-live-test",
    enable_submit_cache: bool = False,
    cache_scope: CacheScopeSelection = "flow",
    cache_expiration_days: int = 7,
    enable_fetch_retry: bool = False,
    enable_result_cache: bool = False,
) -> dict[str, Any]:
    logger = get_run_logger()
    sampler_pubs, estimator_pubs, backend_name = await _prepare_pubs(runtime_block_name)
    logger.warning(
        "This flow submits real Qiskit Runtime jobs to backend %s. "
        "Selected primitive=%s, mode=%s, submit_cache=%s, fetch_retry=%s, "
        "result_cache=%s.",
        backend_name,
        primitive,
        mode,
        enable_submit_cache,
        enable_fetch_retry,
        enable_result_cache,
    )
    submit_sampler = _sampler_submit_task(
        enable_cache=enable_submit_cache,
        cache_expiration_days=cache_expiration_days,
    )
    submit_estimator = _estimator_submit_task(
        enable_cache=enable_submit_cache,
        cache_expiration_days=cache_expiration_days,
    )
    fetch_result = _fetch_task(
        enable_retry=enable_fetch_retry,
        enable_result_cache=enable_result_cache,
    )

    summary: dict[str, Any] = {
        "runtime_block_name": runtime_block_name,
        "backend_name": backend_name,
        "shots": shots,
        "precision": precision,
        "enable_submit_cache": enable_submit_cache,
        "cache_scope": cache_scope,
        "cache_expiration_days": cache_expiration_days,
        "enable_fetch_retry": enable_fetch_retry,
        "enable_result_cache": enable_result_cache,
        "jobs": {},
    }

    if _selected(primitive, "sampler") and _selected(mode, "simple"):
        output = await run_sampler_task(
            sampler_pubs,
            runtime_block_name=runtime_block_name,
            shots=shots,
            artifact_key=f"{artifact_prefix}-sampler-simple",
        )
        summary["jobs"]["sampler_simple"] = _job_summary(output)

    if _selected(primitive, "sampler") and _selected(mode, "robust"):
        input_digest = None
        if enable_submit_cache:
            input_digest = build_qiskit_sampler_input_digest(
                sampler_pubs,
                backend_name=backend_name,
                runtime_block_name=runtime_block_name,
                shots=shots,
                cache_scope=cache_scope,
            )
        reference = await submit_sampler(
            sampler_pubs,
            runtime_block_name=runtime_block_name,
            shots=shots,
            input_digest=input_digest,
        )
        output = await fetch_result(
            runtime_block_name=runtime_block_name,
            job_reference=reference,
            pubs=sampler_pubs,
            artifact_key=f"{artifact_prefix}-sampler-robust",
        )
        summary["jobs"]["sampler_robust_submit"] = _reference_summary(reference)
        summary["jobs"]["sampler_robust_fetch"] = _job_summary(output)

    if _selected(primitive, "estimator") and _selected(mode, "simple"):
        output = await run_estimator_task(
            estimator_pubs,
            runtime_block_name=runtime_block_name,
            precision=precision,
            artifact_key=f"{artifact_prefix}-estimator-simple",
        )
        summary["jobs"]["estimator_simple"] = _job_summary(output)

    if _selected(primitive, "estimator") and _selected(mode, "robust"):
        input_digest = None
        if enable_submit_cache:
            input_digest = build_qiskit_estimator_input_digest(
                estimator_pubs,
                backend_name=backend_name,
                runtime_block_name=runtime_block_name,
                precision=precision,
                cache_scope=cache_scope,
            )
        reference = await submit_estimator(
            estimator_pubs,
            runtime_block_name=runtime_block_name,
            precision=precision,
            input_digest=input_digest,
        )
        output = await fetch_result(
            runtime_block_name=runtime_block_name,
            job_reference=reference,
            pubs=estimator_pubs,
            artifact_key=f"{artifact_prefix}-estimator-robust",
        )
        summary["jobs"]["estimator_robust_submit"] = _reference_summary(reference)
        summary["jobs"]["estimator_robust_fetch"] = _job_summary(output)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit live native Qiskit SamplerV2 and EstimatorV2 jobs through "
            "the qcsc-prefect integration."
        )
    )
    parser.add_argument("--runtime-block", default="ibm-runtime")
    parser.add_argument(
        "--primitive",
        choices=("all", "sampler", "estimator"),
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "simple", "robust"),
        default="all",
        help=(
            "simple uses run_*_task; robust uses submit_*_job_task plus "
            "fetch_qiskit_job_result_task."
        ),
    )
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--precision", type=float, default=0.2)
    parser.add_argument("--artifact-prefix", default="native-qiskit-live-test")
    parser.add_argument(
        "--enable-submit-cache",
        action="store_true",
        help=(
            "Cache submit_*_job_task results. Only robust mode uses submit tasks; "
            "raw result fetches are not cached."
        ),
    )
    parser.add_argument(
        "--cache-scope",
        choices=("flow", "global"),
        default="flow",
        help=(
            "flow scopes input_digest to this Prefect flow; global allows "
            "different flows with the same inputs to share submit cache entries."
        ),
    )
    parser.add_argument("--cache-expiration-days", type=int, default=7)
    parser.add_argument(
        "--enable-fetch-retry",
        action="store_true",
        help="Retry transient failures while fetching an already-submitted job result.",
    )
    parser.add_argument(
        "--enable-result-cache",
        action="store_true",
        help=(
            "Persist fetch_qiskit_job_result_task results with Prefect's "
            "compressed/pickle result cache, keyed by Qiskit job ID."
        ),
    )

    parser.add_argument(
        "--save-runtime-block",
        action="store_true",
        help="Create or overwrite QiskitRuntimeConfig before running the live test.",
    )
    parser.add_argument("--backend-name")
    parser.add_argument("--channel")
    parser.add_argument("--instance")
    parser.add_argument(
        "--token-env",
        default=None,
        help="Environment variable containing the IBM Quantum token when saving a block.",
    )
    parser.add_argument("--account-name")
    parser.add_argument("--filename")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    if args.save_runtime_block:
        if not args.backend_name:
            raise RuntimeError("--backend-name is required with --save-runtime-block.")
        _save_runtime_block(
            block_name=args.runtime_block,
            backend_name=args.backend_name,
            channel=args.channel,
            instance=args.instance,
            token_env=args.token_env,
            account_name=args.account_name,
            filename=args.filename,
        )

    summary = await native_qiskit_primitives_live_test_flow(
        runtime_block_name=args.runtime_block,
        primitive=args.primitive,
        mode=args.mode,
        shots=args.shots,
        precision=args.precision,
        artifact_prefix=args.artifact_prefix,
        enable_submit_cache=args.enable_submit_cache,
        cache_scope=args.cache_scope,
        cache_expiration_days=args.cache_expiration_days,
        enable_fetch_retry=args.enable_fetch_retry,
        enable_result_cache=args.enable_result_cache,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    asyncio.run(_main())
