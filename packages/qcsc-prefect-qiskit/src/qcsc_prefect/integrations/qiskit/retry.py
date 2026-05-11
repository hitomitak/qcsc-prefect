"""Retry helpers for native Qiskit Prefect tasks.

Example:
    robust_fetch = fetch_qiskit_job_result_task.with_options(
        retries=len(qiskit_retry_delays()),
        retry_delay_seconds=qiskit_retry_delays(),
        retry_condition_fn=should_retry_qiskit_fetch_failure,
    )

Submit retries are intentionally conservative because retrying after a job was
accepted by Qiskit Runtime can create duplicate quantum jobs.
"""

from __future__ import annotations

import inspect
from typing import Any


def qiskit_retry_delays() -> list[int]:
    """Return conservative retry delays, in seconds, for Qiskit fetch tasks."""

    return [60, 120, 300, 600, 900]


def is_transient_qiskit_error(exc: BaseException) -> bool:
    """Return whether an exception looks safe to retry for Qiskit fetches."""

    class_name = type(exc).__name__
    module_name = type(exc).__module__
    message = str(exc)
    text = " ".join((class_name, module_name, message)).lower()

    if _contains_any(text, _NON_TRANSIENT_PATTERNS):
        return False

    if isinstance(exc, TimeoutError | ConnectionError | OSError):
        return True

    if _contains_any(text, _TRANSIENT_PATTERNS):
        return True

    if _is_network_exception_module(module_name):
        return True

    return False


def extract_exception_from_state(state: Any) -> BaseException | None:
    """Best-effort extraction of an exception from a failed Prefect state."""

    if isinstance(state, BaseException):
        return state

    for attr_name in ("exception", "error", "exc"):
        exc = _exception_from_value(_safe_getattr(state, attr_name))
        if exc is not None:
            return exc

    result_method = _safe_getattr(state, "result")
    if callable(result_method):
        exc = _exception_from_result_method(result_method)
        if exc is not None:
            return exc

    for attr_name in ("data", "result", "value"):
        exc = _exception_from_value(_safe_getattr(state, attr_name))
        if exc is not None:
            return exc

    return None


def should_retry_qiskit_fetch_failure(_task: Any, _task_run: Any, state: Any) -> bool:
    """Return True when a failed fetch state appears transient and retryable."""

    exc = extract_exception_from_state(state)
    if exc is None:
        return False
    return is_transient_qiskit_error(exc)


def should_retry_qiskit_submit_failure(_task: Any, _task_run: Any, _state: Any) -> bool:
    """Return False by default to avoid duplicate Qiskit Runtime submissions."""

    return False


_NON_TRANSIENT_PATTERNS = (
    "account",
    "auth",
    "bad request",
    "cancelled",
    "canceled",
    "credential",
    "forbidden",
    "invalid",
    "job failed",
    "jobfailure",
    "jobfailed",
    "attributeerror",
    "malformed",
    "keyerror",
    "not authorized",
    "not found",
    "notfound",
    "permission",
    "qiskitjobfailure",
    "unauthorized",
    "unsupported",
    "typeerror",
    "validation",
    "valueerror",
)

_TRANSIENT_PATTERNS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "chunkedencodingerror",
    "connect",
    "connection",
    "dns",
    "gateway",
    "internal server",
    "maintenance",
    "network",
    "pooltimeout",
    "proxyerror",
    "rate limit",
    "ratelimit",
    "readerror",
    "readtimeout",
    "remote disconnected",
    "remoteprotocolerror",
    "requestexception",
    "reset by peer",
    "retryerror",
    "server error",
    "service unavailable",
    "sslerror",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "too many requests",
    "transport",
    "try again",
    "writeerror",
    "writetimeout",
)

_NETWORK_EXCEPTION_MODULE_PREFIXES = (
    "aiohttp.",
    "httpx.",
    "requests.",
    "urllib.",
    "urllib3.",
)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _is_network_exception_module(module_name: str) -> bool:
    return module_name.startswith(_NETWORK_EXCEPTION_MODULE_PREFIXES)


def _safe_getattr(value: Any, attr_name: str) -> Any | None:
    try:
        return getattr(value, attr_name)
    except Exception:
        return None


def _exception_from_value(value: Any) -> BaseException | None:
    if isinstance(value, BaseException):
        return value
    for attr_name in ("exception", "error", "exc", "value"):
        child = _safe_getattr(value, attr_name)
        if isinstance(child, BaseException):
            return child
    return None


def _exception_from_result_method(result_method: Any) -> BaseException | None:
    for kwargs in _result_call_kwargs(result_method):
        try:
            value = result_method(**kwargs)
        except BaseException as exc:
            return exc
        exc = _exception_from_value(value)
        if exc is not None:
            return exc
    return None


def _result_call_kwargs(result_method: Any) -> list[dict[str, bool]]:
    supports_raise_on_failure = _supports_kwarg(result_method, "raise_on_failure")
    if supports_raise_on_failure:
        return [{"raise_on_failure": True}, {"raise_on_failure": False}, {}]
    return [{}]


def _supports_kwarg(callable_obj: Any, kwarg: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == kwarg:
            return True
    return False
