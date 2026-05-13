"""Thin wrapper classes for native Qiskit execution through Prefect tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import timedelta
from typing import Any

from qcsc_prefect.integrations.qiskit.blocks import QiskitRuntimeConfig
from qcsc_prefect.integrations.qiskit.cache import (
    qiskit_estimator_submit_cache_key,
    qiskit_result_fetch_cache_key,
    qiskit_sampler_submit_cache_key,
)
from qcsc_prefect.integrations.qiskit.input_digest import (
    build_qiskit_estimator_input_digest,
    build_qiskit_sampler_input_digest,
)
from qcsc_prefect.integrations.qiskit.retry import (
    qiskit_retry_delays,
    should_retry_qiskit_fetch_failure,
)
from qcsc_prefect.integrations.qiskit.tasks import (
    fetch_qiskit_job_result_task,
    run_estimator_task,
    run_sampler_task,
    submit_estimator_job_task,
    submit_sampler_job_task,
)


class QCSCPrimitiveJob:
    """Job-like handle for split native Qiskit submit/fetch execution.

    The handle is returned by :meth:`QCSCSamplerV2.submit` and
    :meth:`QCSCEstimatorV2.submit`. It stores a serializable Qiskit Runtime job
    reference and fetches the existing job through ``fetch_qiskit_job_result_task``.

    This is not a native Qiskit ``RuntimeJobV2``. It is a small Prefect-facing
    facade that exposes the most common ``job.result()`` style workflow while
    preserving qcsc-prefect metadata and artifact behavior.

    Args:
        job_reference: Serializable reference returned by a submit task.
        runtime_block_name: Optional saved ``QiskitRuntimeConfig`` block name.
        runtime_config: Optional inline runtime configuration object.
        pubs: Native Qiskit pubs used for metadata extraction during fetch.
        artifact_key: Optional Prefect artifact key prefix.
        options: Optional native Qiskit primitive options.
        cache_result: Persist fetched results with Prefect result caching.
        retry_fetch: Retry transient failures while fetching the existing job.
    """

    def __init__(
        self,
        *,
        job_reference: Mapping[str, Any],
        runtime_block_name: str | None,
        runtime_config: QiskitRuntimeConfig | None,
        pubs: Iterable[Any],
        artifact_key: str | None,
        options: Mapping[str, Any] | None,
        cache_result: bool,
        retry_fetch: bool,
    ) -> None:
        self._job_reference = dict(job_reference)
        self._runtime_block_name = runtime_block_name
        self._runtime_config = runtime_config
        self._pubs = list(pubs)
        self._artifact_key = artifact_key
        self._options = _copy_options(options)
        self._cache_result = cache_result
        self._retry_fetch = retry_fetch
        self._output: dict[str, Any] | None = None

    @property
    def job_reference(self) -> dict[str, Any]:
        """Return a copy of the serializable Qiskit Runtime job reference.

        Returns:
            A copy of the submit task job reference dictionary.
        """

        return dict(self._job_reference)

    @property
    def primitive(self) -> str | None:
        """Primitive type recorded in the job reference."""

        value = self._job_reference.get("primitive") or self._job_reference.get("program_type")
        if value is None:
            return None
        return str(value)

    @property
    def backend_name(self) -> str | None:
        """Backend name recorded in the job reference."""

        value = self._job_reference.get("backend_name")
        if value is None:
            return None
        return str(value)

    def job_id(self) -> str | None:
        """Return the native Qiskit Runtime job ID when available.

        Returns:
            The Qiskit Runtime job ID, or ``None`` if the reference does not
            contain one.
        """

        value = self._job_reference.get("job_id")
        if value is None:
            return None
        return str(value)

    async def output(self) -> dict[str, Any]:
        """Fetch and return the full qcsc-prefect task output dictionary.

        The first call fetches the native Qiskit Runtime job by ID. Later calls
        on the same handle reuse the in-memory output. When ``cache_result`` is
        enabled, Prefect can also restore the output from result storage across
        reruns that use the same Qiskit job ID.

        Returns:
            The structured dictionary returned by
            ``fetch_qiskit_job_result_task``. It includes ``result`` and
            ``metadata`` entries.
        """

        if self._output is None:
            fetch_task = _fetch_task(
                cache_result=self._cache_result,
                retry_fetch=self._retry_fetch,
            )
            self._output = await fetch_task(
                **_runtime_kwargs(self._runtime_block_name, self._runtime_config),
                job_reference=self._job_reference,
                pubs=self._pubs,
                artifact_key=self._artifact_key,
                options=self._options,
            )
        return self._output

    async def result(self) -> Any:
        """Fetch and return the native Qiskit primitive result object.

        Returns:
            The native Qiskit primitive result stored in ``output()["result"]``.
        """

        output = await self.output()
        return output["result"]


class QCSCSamplerV2:
    """Thin SamplerV2-style facade backed by native Qiskit Prefect tasks.

    ``QCSCSamplerV2`` keeps runtime configuration and delegates execution to the
    Native Qiskit Prefect task API. It does not subclass or reimplement Qiskit's
    ``SamplerV2``.

    Args:
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` Prefect
            block. Mutually exclusive with ``runtime_config``.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.
        options: Default native Qiskit Sampler options. ``run`` and ``submit``
            can override this value.
        backend_name: Optional backend name used only for submit cache input
            digest construction when it cannot be inferred from
            ``runtime_config``.
    """

    def __init__(
        self,
        *,
        runtime_block_name: str | None = None,
        runtime_config: QiskitRuntimeConfig | None = None,
        options: Mapping[str, Any] | None = None,
        backend_name: str | None = None,
    ) -> None:
        _validate_runtime_source(runtime_block_name, runtime_config)
        self.runtime_block_name = runtime_block_name
        self.runtime_config = runtime_config
        self.options = _copy_options(options)
        self.backend_name = backend_name or _config_backend_name(runtime_config)

    async def run(
        self,
        pubs: Iterable[Any],
        *,
        shots: int | None = None,
        artifact_key: str | None = None,
        options: Mapping[str, Any] | None = None,
        robust: bool = True,
        cache_submit: bool = False,
        cache_result: bool = False,
        retry_fetch: bool = False,
        cache_scope: str = "flow",
        cache_namespace: str | None = None,
        cache_expiration: timedelta | None = None,
        input_digest: str | None = None,
    ) -> dict[str, Any]:
        """Run Sampler pubs through the configured Prefect integration path.

        With ``robust=True`` this method submits a native Qiskit Runtime job,
        fetches the same job by ID, records artifacts and returns the full task
        output. With ``robust=False`` it delegates to ``run_sampler_task``.

        Args:
            pubs: Native Qiskit Sampler pubs.
            shots: Optional number of shots passed to ``SamplerV2.run``.
            artifact_key: Optional Prefect artifact key prefix.
            options: Optional Sampler options for this call. Replaces wrapper
                default options when provided.
            robust: Use split submit/fetch mode. Required for cache and retry
                flags.
            cache_submit: Cache the submit task result to avoid duplicate job
                submission for the same input digest.
            cache_result: Persist fetched results with Prefect result caching,
                keyed by Qiskit Runtime job ID.
            retry_fetch: Retry transient failures while fetching the existing
                job result.
            cache_scope: Input digest scope used when ``cache_submit=True`` and
                ``input_digest`` is not provided.
            cache_namespace: Optional namespace included in the input digest.
            cache_expiration: Optional Prefect cache expiration for submit
                caching.
            input_digest: Optional precomputed digest for submit caching.

        Returns:
            The structured dictionary returned by the underlying run or fetch
            task. The native Qiskit result is available as ``output["result"]``.
        """

        pub_list = list(pubs)
        resolved_options = _resolve_options(self.options, options)
        runtime_kwargs = _runtime_kwargs(self.runtime_block_name, self.runtime_config)

        if not robust:
            _reject_robust_only_options(
                cache_submit=cache_submit,
                cache_result=cache_result,
                retry_fetch=retry_fetch,
            )
            return await run_sampler_task(
                pub_list,
                **runtime_kwargs,
                shots=shots,
                artifact_key=artifact_key,
                options=resolved_options,
                input_digest=input_digest,
            )

        job = await self.submit(
            pub_list,
            shots=shots,
            artifact_key=artifact_key,
            options=resolved_options,
            cache_submit=cache_submit,
            cache_result=cache_result,
            retry_fetch=retry_fetch,
            cache_scope=cache_scope,
            cache_namespace=cache_namespace,
            cache_expiration=cache_expiration,
            input_digest=input_digest,
        )
        return await job.output()

    async def submit(
        self,
        pubs: Iterable[Any],
        *,
        shots: int | None = None,
        artifact_key: str | None = None,
        options: Mapping[str, Any] | None = None,
        cache_submit: bool = False,
        cache_result: bool = False,
        retry_fetch: bool = False,
        cache_scope: str = "flow",
        cache_namespace: str | None = None,
        cache_expiration: timedelta | None = None,
        input_digest: str | None = None,
    ) -> QCSCPrimitiveJob:
        """Submit a Sampler job and return a job-like fetch handle.

        Args:
            pubs: Native Qiskit Sampler pubs.
            shots: Optional number of shots passed to ``SamplerV2.run``.
            artifact_key: Optional Prefect artifact key prefix used when
                fetching results.
            options: Optional Sampler options for this call.
            cache_submit: Cache the submit task result to avoid duplicate job
                submission for the same input digest.
            cache_result: Persist fetched results with Prefect result caching
                when ``QCSCPrimitiveJob.result`` or ``output`` is called.
            retry_fetch: Retry transient failures during ``result`` or
                ``output``.
            cache_scope: Input digest scope used when ``cache_submit=True`` and
                ``input_digest`` is not provided.
            cache_namespace: Optional namespace included in the input digest.
            cache_expiration: Optional Prefect cache expiration for submit
                caching.
            input_digest: Optional precomputed digest for submit caching.

        Returns:
            A ``QCSCPrimitiveJob`` whose ``result`` method fetches the native
            Qiskit primitive result.
        """

        pub_list = list(pubs)
        resolved_options = _resolve_options(self.options, options)
        runtime_kwargs = _runtime_kwargs(self.runtime_block_name, self.runtime_config)
        resolved_digest = input_digest
        if cache_submit and resolved_digest is None:
            resolved_digest = build_qiskit_sampler_input_digest(
                pub_list,
                backend_name=self.backend_name,
                runtime_block_name=self.runtime_block_name,
                shots=shots,
                options=resolved_options,
                cache_scope=cache_scope,
                cache_namespace=cache_namespace,
            )

        submit_task = _submit_task(
            submit_sampler_job_task,
            cache_key_fn=qiskit_sampler_submit_cache_key,
            cache_submit=cache_submit,
            cache_expiration=cache_expiration,
        )
        job_reference = await submit_task(
            pub_list,
            **runtime_kwargs,
            shots=shots,
            options=resolved_options,
            input_digest=resolved_digest,
        )

        return QCSCPrimitiveJob(
            job_reference=job_reference,
            pubs=pub_list,
            artifact_key=artifact_key,
            options=resolved_options,
            runtime_block_name=self.runtime_block_name,
            runtime_config=self.runtime_config,
            cache_result=cache_result,
            retry_fetch=retry_fetch,
        )


class QCSCEstimatorV2:
    """Thin EstimatorV2-style facade backed by native Qiskit Prefect tasks.

    ``QCSCEstimatorV2`` mirrors :class:`QCSCSamplerV2` for Estimator pubs. It
    delegates to native Qiskit Runtime task helpers and does not subclass or
    reimplement Qiskit's ``EstimatorV2``.

    Args:
        runtime_block_name: Name of a saved ``QiskitRuntimeConfig`` Prefect
            block. Mutually exclusive with ``runtime_config``.
        runtime_config: Inline runtime configuration. Mutually exclusive with
            ``runtime_block_name``.
        options: Default native Qiskit Estimator options. ``run`` and
            ``submit`` can override this value.
        backend_name: Optional backend name used only for submit cache input
            digest construction when it cannot be inferred from
            ``runtime_config``.
    """

    def __init__(
        self,
        *,
        runtime_block_name: str | None = None,
        runtime_config: QiskitRuntimeConfig | None = None,
        options: Mapping[str, Any] | None = None,
        backend_name: str | None = None,
    ) -> None:
        _validate_runtime_source(runtime_block_name, runtime_config)
        self.runtime_block_name = runtime_block_name
        self.runtime_config = runtime_config
        self.options = _copy_options(options)
        self.backend_name = backend_name or _config_backend_name(runtime_config)

    async def run(
        self,
        pubs: Iterable[Any],
        *,
        precision: float | None = None,
        artifact_key: str | None = None,
        options: Mapping[str, Any] | None = None,
        robust: bool = True,
        cache_submit: bool = False,
        cache_result: bool = False,
        retry_fetch: bool = False,
        cache_scope: str = "flow",
        cache_namespace: str | None = None,
        cache_expiration: timedelta | None = None,
        input_digest: str | None = None,
    ) -> dict[str, Any]:
        """Run Estimator pubs through the configured Prefect integration path.

        With ``robust=True`` this method submits a native Qiskit Runtime job,
        fetches the same job by ID, records artifacts and returns the full task
        output. With ``robust=False`` it delegates to ``run_estimator_task``.

        Args:
            pubs: Native Qiskit Estimator pubs.
            precision: Optional target precision passed to ``EstimatorV2.run``.
            artifact_key: Optional Prefect artifact key prefix.
            options: Optional Estimator options for this call. Replaces wrapper
                default options when provided.
            robust: Use split submit/fetch mode. Required for cache and retry
                flags.
            cache_submit: Cache the submit task result to avoid duplicate job
                submission for the same input digest.
            cache_result: Persist fetched results with Prefect result caching,
                keyed by Qiskit Runtime job ID.
            retry_fetch: Retry transient failures while fetching the existing
                job result.
            cache_scope: Input digest scope used when ``cache_submit=True`` and
                ``input_digest`` is not provided.
            cache_namespace: Optional namespace included in the input digest.
            cache_expiration: Optional Prefect cache expiration for submit
                caching.
            input_digest: Optional precomputed digest for submit caching.

        Returns:
            The structured dictionary returned by the underlying run or fetch
            task. The native Qiskit result is available as ``output["result"]``.
        """

        pub_list = list(pubs)
        resolved_options = _resolve_options(self.options, options)
        runtime_kwargs = _runtime_kwargs(self.runtime_block_name, self.runtime_config)

        if not robust:
            _reject_robust_only_options(
                cache_submit=cache_submit,
                cache_result=cache_result,
                retry_fetch=retry_fetch,
            )
            return await run_estimator_task(
                pub_list,
                **runtime_kwargs,
                precision=precision,
                artifact_key=artifact_key,
                options=resolved_options,
                input_digest=input_digest,
            )

        job = await self.submit(
            pub_list,
            precision=precision,
            artifact_key=artifact_key,
            options=resolved_options,
            cache_submit=cache_submit,
            cache_result=cache_result,
            retry_fetch=retry_fetch,
            cache_scope=cache_scope,
            cache_namespace=cache_namespace,
            cache_expiration=cache_expiration,
            input_digest=input_digest,
        )
        return await job.output()

    async def submit(
        self,
        pubs: Iterable[Any],
        *,
        precision: float | None = None,
        artifact_key: str | None = None,
        options: Mapping[str, Any] | None = None,
        cache_submit: bool = False,
        cache_result: bool = False,
        retry_fetch: bool = False,
        cache_scope: str = "flow",
        cache_namespace: str | None = None,
        cache_expiration: timedelta | None = None,
        input_digest: str | None = None,
    ) -> QCSCPrimitiveJob:
        """Submit an Estimator job and return a job-like fetch handle.

        Args:
            pubs: Native Qiskit Estimator pubs.
            precision: Optional target precision passed to ``EstimatorV2.run``.
            artifact_key: Optional Prefect artifact key prefix used when
                fetching results.
            options: Optional Estimator options for this call.
            cache_submit: Cache the submit task result to avoid duplicate job
                submission for the same input digest.
            cache_result: Persist fetched results with Prefect result caching
                when ``QCSCPrimitiveJob.result`` or ``output`` is called.
            retry_fetch: Retry transient failures during ``result`` or
                ``output``.
            cache_scope: Input digest scope used when ``cache_submit=True`` and
                ``input_digest`` is not provided.
            cache_namespace: Optional namespace included in the input digest.
            cache_expiration: Optional Prefect cache expiration for submit
                caching.
            input_digest: Optional precomputed digest for submit caching.

        Returns:
            A ``QCSCPrimitiveJob`` whose ``result`` method fetches the native
            Qiskit primitive result.
        """

        pub_list = list(pubs)
        resolved_options = _resolve_options(self.options, options)
        runtime_kwargs = _runtime_kwargs(self.runtime_block_name, self.runtime_config)
        resolved_digest = input_digest
        if cache_submit and resolved_digest is None:
            resolved_digest = build_qiskit_estimator_input_digest(
                pub_list,
                backend_name=self.backend_name,
                runtime_block_name=self.runtime_block_name,
                precision=precision,
                options=resolved_options,
                cache_scope=cache_scope,
                cache_namespace=cache_namespace,
            )

        submit_task = _submit_task(
            submit_estimator_job_task,
            cache_key_fn=qiskit_estimator_submit_cache_key,
            cache_submit=cache_submit,
            cache_expiration=cache_expiration,
        )
        job_reference = await submit_task(
            pub_list,
            **runtime_kwargs,
            precision=precision,
            options=resolved_options,
            input_digest=resolved_digest,
        )

        return QCSCPrimitiveJob(
            job_reference=job_reference,
            pubs=pub_list,
            artifact_key=artifact_key,
            options=resolved_options,
            runtime_block_name=self.runtime_block_name,
            runtime_config=self.runtime_config,
            cache_result=cache_result,
            retry_fetch=retry_fetch,
        )


def _validate_runtime_source(
    runtime_block_name: str | None,
    runtime_config: QiskitRuntimeConfig | None,
) -> None:
    if runtime_block_name is not None and runtime_config is not None:
        raise ValueError("Pass either runtime_block_name or runtime_config, not both.")
    if runtime_block_name is None and runtime_config is None:
        raise ValueError("Either runtime_block_name or runtime_config is required.")


def _runtime_kwargs(
    runtime_block_name: str | None,
    runtime_config: QiskitRuntimeConfig | None,
) -> dict[str, Any]:
    if runtime_config is not None:
        return {"runtime_config": runtime_config}
    return {"runtime_block_name": runtime_block_name}


def _config_backend_name(runtime_config: QiskitRuntimeConfig | None) -> str | None:
    if runtime_config is None:
        return None
    value = getattr(runtime_config, "backend_name", None)
    if value is None:
        return None
    return str(value)


def _copy_options(options: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if options is None:
        return None
    return dict(options)


def _resolve_options(
    default_options: Mapping[str, Any] | None,
    run_options: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if run_options is not None:
        return dict(run_options)
    return _copy_options(default_options)


def _submit_task(
    submit_task: Any,
    *,
    cache_key_fn: Any,
    cache_submit: bool,
    cache_expiration: timedelta | None,
) -> Any:
    if not cache_submit:
        return submit_task

    task_options: dict[str, Any] = {
        "cache_key_fn": cache_key_fn,
        "persist_result": True,
    }
    if cache_expiration is not None:
        task_options["cache_expiration"] = cache_expiration
    return submit_task.with_options(**task_options)


def _fetch_task(*, cache_result: bool, retry_fetch: bool) -> Any:
    task_options: dict[str, Any] = {}
    if cache_result:
        task_options.update(
            {
                "cache_key_fn": qiskit_result_fetch_cache_key,
                "persist_result": True,
                "result_serializer": "compressed/pickle",
            }
        )
    if retry_fetch:
        task_options.update(
            {
                "retries": len(qiskit_retry_delays()),
                "retry_delay_seconds": qiskit_retry_delays(),
                "retry_condition_fn": should_retry_qiskit_fetch_failure,
            }
        )
    if not task_options:
        return fetch_qiskit_job_result_task
    return fetch_qiskit_job_result_task.with_options(**task_options)


def _reject_robust_only_options(
    *,
    cache_submit: bool,
    cache_result: bool,
    retry_fetch: bool,
) -> None:
    if cache_submit or cache_result or retry_fetch:
        raise ValueError(
            "cache_submit, cache_result, and retry_fetch require robust=True "
            "because they operate on split submit/fetch tasks."
        )
