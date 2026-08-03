from __future__ import annotations


class SubmitError(RuntimeError):
    """Raised when bulk scheduler submission fails irrecoverably."""


class DuplicateJobKeyError(SubmitError):
    """Raised when a bulk job key already refers to a submitted or terminal job."""


class SpecHashMismatchError(SubmitError):
    """Raised when one job key is reused for a different resolved job spec."""

    def __init__(
        self,
        *,
        job_key: str,
        stored_spec_hash: str,
        incoming_spec_hash: str,
    ) -> None:
        self.job_key = job_key
        self.stored_spec_hash = stored_spec_hash
        self.incoming_spec_hash = incoming_spec_hash
        super().__init__(
            f"Bulk job key {job_key!r} has immutable spec hash "
            f"{stored_spec_hash!r}, not {incoming_spec_hash!r}. "
            "Use a new job_key for a changed command, input, environment, or resource spec."
        )


class QueueFullError(SubmitError):
    """Raised when a scheduler rejects submission because queue capacity is full."""


class TemporarySubmitError(SubmitError):
    """Raised when a scheduler submission failure should be retried later."""


class CancellationRequestedError(SubmitError):
    """Raised when a durable cancellation intent forbids a new submission."""

    def __init__(self, *, job_key: str) -> None:
        self.job_key = str(job_key)
        super().__init__(
            f"Bulk job {self.job_key!r} has a durable cancellation request; "
            "scheduler submission is forbidden."
        )


class SubmitOutcomeUnknownError(SubmitError):
    """Raised when submission may have succeeded and must not be repeated."""

    submit_outcome_unknown = True


class RecoveryPending(SubmitError):
    """Raised when a prepared job is still inside its scheduler-visibility grace."""

    def __init__(self, *, job_key: str, retry_after_seconds: float) -> None:
        self.job_key = job_key
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        super().__init__(
            f"Bulk job {job_key!r} remains PREPARED while Slurm identity visibility "
            f"is within grace; reconcile again after {self.retry_after_seconds:g} seconds."
        )


class OperatorActionRequired(SubmitError):
    """Raised when automatic recovery has failed closed in a durable hold."""

    def __init__(self, *, job_keys: list[str], reason: str) -> None:
        self.job_keys = tuple(dict.fromkeys(str(job_key) for job_key in job_keys))
        self.reason = str(reason)
        super().__init__(
            "Operator action is required for bulk job(s) "
            f"{', '.join(repr(job_key) for job_key in self.job_keys)}: {self.reason}"
        )


class SchedulerIdentityMismatchError(SpecHashMismatchError):
    """Raised when a scheduler candidate conflicts with stored immutable identity."""

    def __init__(self, *, job_key: str, stored_spec_hash: str) -> None:
        self.job_key = job_key
        self.stored_spec_hash = stored_spec_hash
        self.incoming_spec_hash = "<scheduler-identity-mismatch>"
        SubmitError.__init__(
            self,
            f"Slurm candidate identity does not match the immutable identity for "
            f"bulk job {job_key!r} with spec hash {stored_spec_hash!r}. "
            "Automatic attach is forbidden; operator reconciliation is required.",
        )
