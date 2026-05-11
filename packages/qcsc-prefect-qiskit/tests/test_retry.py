from __future__ import annotations

from qcsc_prefect.integrations.qiskit.retry import (
    extract_exception_from_state,
    is_transient_qiskit_error,
    qiskit_retry_delays,
    should_retry_qiskit_fetch_failure,
    should_retry_qiskit_submit_failure,
)


class _AuthenticationError(Exception):
    __module__ = "qiskit_ibm_runtime.exceptions"


class _JobNotFoundError(Exception):
    __module__ = "qiskit_ibm_runtime.exceptions"


class _HttpxTransportError(Exception):
    __module__ = "httpx"


class _StateRaises:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def result(self, *, raise_on_failure: bool = True):
        if raise_on_failure:
            raise self.exc
        return self.exc


class _StateWithExceptionAttr:
    def __init__(self, exc: BaseException) -> None:
        self.exception = exc


class _UnknownState:
    pass


def test_qiskit_retry_delays_are_conservative_defaults():
    assert qiskit_retry_delays() == [60, 120, 300, 600, 900]


def test_timeout_error_is_transient():
    assert is_transient_qiskit_error(TimeoutError("timed out")) is True


def test_connection_error_is_transient():
    assert is_transient_qiskit_error(ConnectionError("connection reset")) is True


def test_os_error_is_transient():
    assert is_transient_qiskit_error(OSError("network unreachable")) is True


def test_network_exception_module_is_transient():
    assert is_transient_qiskit_error(_HttpxTransportError("transport failed")) is True


def test_value_error_is_not_transient():
    assert is_transient_qiskit_error(ValueError("invalid circuit")) is False


def test_authentication_like_error_is_not_transient():
    exc = _AuthenticationError("Authentication failed for IBM Quantum credentials")
    assert is_transient_qiskit_error(exc) is False


def test_job_not_found_like_error_is_not_transient():
    exc = _JobNotFoundError("Qiskit Runtime job job-123 was not found")
    assert is_transient_qiskit_error(exc) is False


def test_submit_retry_helper_returns_false():
    state = _StateRaises(TimeoutError("timed out"))
    assert should_retry_qiskit_submit_failure(None, None, state) is False


def test_fetch_retry_helper_returns_true_for_transient_failure_state():
    state = _StateRaises(TimeoutError("timed out"))
    assert should_retry_qiskit_fetch_failure(None, None, state) is True


def test_fetch_retry_helper_returns_false_for_non_transient_failure_state():
    state = _StateRaises(ValueError("invalid input"))
    assert should_retry_qiskit_fetch_failure(None, None, state) is False


def test_extract_exception_from_state_reads_common_exception_attrs():
    exc = ConnectionError("connection reset")
    assert extract_exception_from_state(_StateWithExceptionAttr(exc)) is exc


def test_helpers_do_not_crash_on_unknown_state_objects():
    state = _UnknownState()
    assert extract_exception_from_state(state) is None
    assert should_retry_qiskit_fetch_failure(None, None, state) is False
    assert should_retry_qiskit_submit_failure(None, None, state) is False
