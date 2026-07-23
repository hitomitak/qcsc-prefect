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
