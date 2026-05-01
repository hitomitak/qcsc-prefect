"""Prefect task utilities for native Qiskit execution."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from typing import Any

from anyio.to_thread import run_sync
from prefect import get_run_logger, task
from qcsc_prefect.integrations.qiskit.artifacts import (
    create_qiskit_estimator_result_artifact,
    create_qiskit_execution_markdown_artifact,
    create_qiskit_sampler_result_artifact,
)
from qcsc_prefect.integrations.qiskit.blocks import QiskitRuntimeConfig
from qcsc_prefect.integrations.qiskit.metadata import collect_qiskit_execution_metadata


def _sampler_class():
    from qiskit_ibm_runtime import SamplerV2

    return SamplerV2


def _estimator_class():
    from qiskit_ibm_runtime import EstimatorV2

    return EstimatorV2


class QiskitSamplerTaskError(RuntimeError):
    """Raised when the native Qiskit Sampler task cannot complete."""


class QiskitEstimatorTaskError(RuntimeError):
    """Raised when the native Qiskit Estimator task cannot complete."""


async def _resolve_loaded_block(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _load_runtime_config(
    runtime_block_name: str,
    *,
    error_cls: type[RuntimeError] = QiskitSamplerTaskError,
) -> Any:
    try:
        return await _resolve_loaded_block(QiskitRuntimeConfig.load(runtime_block_name))
    except Exception as exc:
        raise error_cls(
            "Failed to load QiskitRuntimeConfig block "
            f"{runtime_block_name!r} ({type(exc).__name__})."
        ) from None


def _logger() -> logging.Logger:
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


def _job_id(job: Any) -> str | None:
    value = getattr(job, "job_id", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return str(value)


def _backend_name(backend: Any) -> str | None:
    for attr in ("name", "backend_name"):
        value = getattr(backend, attr, None)
        if callable(value):
            value = value()
        if value is not None:
            return str(value)
    if isinstance(backend, str):
        return backend
    return None


def _metadata_options(
    options: dict[str, Any] | None,
    *,
    shots: int | None = None,
    precision: float | None = None,
) -> dict[str, Any]:
    metadata_options: dict[str, Any] = dict(options or {})
    if shots is not None or precision is not None:
        params = metadata_options.get("params")
        if isinstance(params, dict):
            params = dict(params)
        else:
            params = {}
        if shots is not None:
            params.setdefault("shots", shots)
        if precision is not None:
            params.setdefault("precision", precision)
        metadata_options["params"] = params
    return metadata_options


def _create_sampler(*, backend: Any, backend_name: str, options: dict[str, Any] | None) -> Any:
    sampler_kwargs: dict[str, Any] = {"mode": backend}
    if options is not None:
        sampler_kwargs["options"] = options

    try:
        return _sampler_class()(**sampler_kwargs)
    except Exception as exc:
        raise QiskitSamplerTaskError(
            "Failed to create native SamplerV2 "
            f"for backend {backend_name!r} ({type(exc).__name__})."
        ) from None


def _create_estimator(*, backend: Any, backend_name: str, options: dict[str, Any] | None) -> Any:
    estimator_kwargs: dict[str, Any] = {"mode": backend}
    if options is not None:
        estimator_kwargs["options"] = options

    try:
        return _estimator_class()(**estimator_kwargs)
    except Exception as exc:
        raise QiskitEstimatorTaskError(
            "Failed to create native EstimatorV2 "
            f"for backend {backend_name!r} ({type(exc).__name__})."
        ) from None


def _submit_sampler_job(
    *,
    sampler: Any,
    pubs: list[Any],
    shots: int | None,
    backend_name: str,
) -> Any:
    try:
        return sampler.run(pubs, shots=shots)
    except Exception as exc:
        raise QiskitSamplerTaskError(
            "Failed to submit native Qiskit SamplerV2 job "
            f"to backend {backend_name!r} ({type(exc).__name__})."
        ) from None


def _submit_estimator_job(
    *,
    estimator: Any,
    pubs: list[Any],
    precision: float | None,
    backend_name: str,
) -> Any:
    try:
        return estimator.run(pubs, precision=precision)
    except Exception as exc:
        raise QiskitEstimatorTaskError(
            "Failed to submit native Qiskit EstimatorV2 job "
            f"to backend {backend_name!r} ({type(exc).__name__})."
        ) from None


async def _wait_for_primitive_result(
    *,
    job: Any,
    job_id: str | None,
    backend_name: str,
    primitive_name: str,
    error_cls: type[RuntimeError],
) -> Any:
    try:
        return await run_sync(job.result)
    except Exception as exc:
        raise error_cls(
            f"Failed while waiting for native Qiskit {primitive_name} job result "
            f"(job_id={job_id!r}, backend={backend_name!r}, {type(exc).__name__})."
        ) from None


@task(name="run-qiskit-sampler")
async def run_sampler_task(
    pubs: Iterable[Any],
    runtime_block_name: str,
    shots: int | None = None,
    artifact_key: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run native Qiskit Runtime ``SamplerV2`` inside a Prefect task."""

    logger = _logger()
    runtime_config = await _load_runtime_config(runtime_block_name)
    backend = runtime_config.get_backend()
    backend_name = _backend_name(backend) or runtime_config.backend_name

    sampler = _create_sampler(backend=backend, backend_name=backend_name, options=options)
    pub_list = list(pubs)
    job = _submit_sampler_job(
        sampler=sampler,
        pubs=pub_list,
        shots=shots,
        backend_name=backend_name,
    )
    job_id = _job_id(job)
    logger.info(
        "Submitted Qiskit SamplerV2 job %s to backend %s.",
        job_id or "<unknown>",
        backend_name,
    )

    result = await _wait_for_primitive_result(
        job=job,
        job_id=job_id,
        backend_name=backend_name,
        primitive_name="SamplerV2",
        error_cls=QiskitSamplerTaskError,
    )
    metadata = collect_qiskit_execution_metadata(
        job=job,
        pubs=pub_list,
        result=result,
        options=_metadata_options(options, shots=shots),
        resource=backend_name,
        program_type="sampler",
    )
    if metadata.job_id is None:
        metadata.job_id = job_id

    await create_qiskit_execution_markdown_artifact(
        metadata,
        key=artifact_key or "qiskit-sampler-summary",
    )
    await create_qiskit_sampler_result_artifact(
        result,
        key=f"{artifact_key or 'qiskit-sampler-summary'}-result",
    )

    return {
        "primitive": "sampler",
        "backend_name": backend_name,
        "job_id": metadata.job_id,
        "shots": shots,
        "result": result,
        "metadata": metadata,
    }


@task(name="run-qiskit-estimator")
async def run_estimator_task(
    pubs: Iterable[Any],
    runtime_block_name: str,
    precision: float | None = None,
    artifact_key: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run native Qiskit Runtime ``EstimatorV2`` inside a Prefect task."""

    logger = _logger()
    runtime_config = await _load_runtime_config(
        runtime_block_name,
        error_cls=QiskitEstimatorTaskError,
    )
    backend = runtime_config.get_backend()
    backend_name = _backend_name(backend) or runtime_config.backend_name

    estimator = _create_estimator(backend=backend, backend_name=backend_name, options=options)
    pub_list = list(pubs)
    job = _submit_estimator_job(
        estimator=estimator,
        pubs=pub_list,
        precision=precision,
        backend_name=backend_name,
    )
    job_id = _job_id(job)
    logger.info(
        "Submitted Qiskit EstimatorV2 job %s to backend %s.",
        job_id or "<unknown>",
        backend_name,
    )

    result = await _wait_for_primitive_result(
        job=job,
        job_id=job_id,
        backend_name=backend_name,
        primitive_name="EstimatorV2",
        error_cls=QiskitEstimatorTaskError,
    )
    metadata = collect_qiskit_execution_metadata(
        job=job,
        pubs=pub_list,
        result=result,
        options=_metadata_options(options, precision=precision),
        resource=backend_name,
        program_type="estimator",
    )
    if metadata.job_id is None:
        metadata.job_id = job_id

    await create_qiskit_execution_markdown_artifact(
        metadata,
        key=artifact_key or "qiskit-estimator-summary",
    )
    await create_qiskit_estimator_result_artifact(
        result,
        key=f"{artifact_key or 'qiskit-estimator-summary'}-result",
    )

    return {
        "primitive": "estimator",
        "backend_name": backend_name,
        "job_id": metadata.job_id,
        "precision": precision,
        "result": result,
        "metadata": metadata,
    }
