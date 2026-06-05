from __future__ import annotations


class SubmitError(RuntimeError):
    """Raised when bulk scheduler submission fails irrecoverably."""


class DuplicateJobKeyError(SubmitError):
    """Raised when a bulk job key already refers to a submitted or terminal job."""


class QueueFullError(SubmitError):
    """Raised when a scheduler rejects submission because queue capacity is full."""


class TemporarySubmitError(SubmitError):
    """Raised when a scheduler submission failure should be retried later."""
