from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
from qcsc_prefect.integrations.qiskit import tasks as tasks_mod
from qcsc_prefect.integrations.qiskit.durable import (
    QISKIT_MAX_JOB_TAGS,
    QiskitIdentityRecoveryRuntime,
    QiskitJobIdentity,
    QiskitOperatorActionRequired,
    QiskitRecoveryPending,
    QiskitSpecHashMismatchError,
    QiskitSubmissionJournal,
    QiskitSubmissionStatus,
    build_qiskit_submission_tag,
    with_qiskit_submission_tags,
)
from qcsc_prefect.integrations.qiskit.tasks import (
    submit_estimator_job_task,
    submit_sampler_job_task,
)


class _Backend:
    name = "ibm_kawasaki"

    def __init__(self, service: "_Service") -> None:
        self.service = service


class _Job:
    def __init__(
        self,
        job_id: str,
        *,
        backend: _Backend,
        tags: list[str],
        creation_date: datetime | None = None,
    ) -> None:
        self._job_id = job_id
        self._backend = backend
        self._tags = list(tags)
        self._creation_date = creation_date or datetime.now(timezone.utc)

    def job_id(self) -> str:
        return self._job_id

    def backend(self) -> _Backend:
        return self._backend

    def tags(self) -> list[str]:
        return list(self._tags)

    def creation_date(self) -> datetime:
        return self._creation_date


class _Service:
    def __init__(self) -> None:
        self.search_results: list[_Job] = []
        self.jobs_calls: list[dict[str, Any]] = []

    def jobs(self, **kwargs: Any) -> list[_Job]:
        self.jobs_calls.append(dict(kwargs))
        return list(self.search_results)


class _RuntimeConfig:
    backend_name = "ibm_kawasaki"

    def __init__(self) -> None:
        self.service = _Service()
        self.backend = _Backend(self.service)
        self.backend_calls = 0
        self.service_calls = 0

    def get_backend(self) -> _Backend:
        self.backend_calls += 1
        return self.backend

    def get_service(self) -> _Service:
        self.service_calls += 1
        return self.service


class _Sampler:
    instances: list["_Sampler"] = []

    def __init__(self, *, mode: _Backend, options: dict[str, Any] | None = None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls: list[dict[str, Any]] = []
        self.instances.append(self)

    def run(self, pubs: list[Any], *, shots: int | None = None) -> _Job:
        self.run_calls.append({"pubs": pubs, "shots": shots})
        tags = list((self.options or {}).get("environment", {}).get("job_tags", []))
        job = _Job(
            f"runtime-job-{len(self.mode.service.search_results) + 1}",
            backend=self.mode,
            tags=tags,
        )
        self.mode.service.search_results.append(job)
        return job


class _Estimator:
    instances: list["_Estimator"] = []

    def __init__(self, *, mode: _Backend, options: dict[str, Any] | None = None) -> None:
        self.mode = mode
        self.options = options
        self.run_calls: list[dict[str, Any]] = []
        self.instances.append(self)

    def run(self, pubs: list[Any], *, precision: float | None = None) -> _Job:
        self.run_calls.append({"pubs": pubs, "precision": precision})
        tags = list((self.options or {}).get("environment", {}).get("job_tags", []))
        job = _Job(
            "estimator-runtime-job-1",
            backend=self.mode,
            tags=tags,
        )
        self.mode.service.search_results.append(job)
        return job


def _patch_sampler(monkeypatch) -> None:
    _Sampler.instances = []
    monkeypatch.setattr(tasks_mod, "_sampler_class", lambda: _Sampler)


def _patch_estimator(monkeypatch) -> None:
    _Estimator.instances = []
    monkeypatch.setattr(tasks_mod, "_estimator_class", lambda: _Estimator)


def _durable_submit(
    *,
    runtime_config: _RuntimeConfig,
    journal_path: Path,
    submission_key: str = "submission-1",
    spec_hash: str = "spec-v1:abc",
    recovery_grace_seconds: float = 120.0,
):
    return submit_sampler_job_task.fn(
        ["pub-0"],
        runtime_config=runtime_config,
        shots=100,
        options={"resilience_level": 1},
        input_digest="input-digest",
        submission_key=submission_key,
        spec_hash=spec_hash,
        journal_path=journal_path,
        submission_tags=["campaign-safe"],
        recovery_grace_seconds=recovery_grace_seconds,
    )


def test_journal_prepared_claim_is_atomic(tmp_path: Path):
    path = tmp_path / "qiskit.sqlite"
    first = QiskitSubmissionJournal(path)
    second = QiskitSubmissionJournal(path)
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    barrier = Barrier(2)

    def claim(journal: QiskitSubmissionJournal) -> bool:
        barrier.wait()
        _, claimed = journal.prepare(
            submission_key="submission-1",
            spec_hash="spec-v1:abc",
            stable_tag=stable_tag,
        )
        return claimed

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, [first, second]))

    assert sorted(results) == [False, True]
    record = first.get("submission-1")
    assert record is not None
    assert record.status == QiskitSubmissionStatus.PREPARED


def test_journal_rejects_spec_change_without_mutation(tmp_path: Path):
    journal = QiskitSubmissionJournal(tmp_path / "qiskit.sqlite")
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    journal.prepare(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
        stable_tag=stable_tag,
    )

    with pytest.raises(QiskitSpecHashMismatchError):
        journal.prepare(
            submission_key="submission-1",
            spec_hash="spec-v1:different",
            stable_tag=build_qiskit_submission_tag(
                submission_key="submission-1",
                spec_hash="spec-v1:different",
            ),
        )

    record = journal.get("submission-1")
    assert record is not None
    assert record.spec_hash == "spec-v1:abc"
    assert record.stable_tag == stable_tag
    assert record.status == QiskitSubmissionStatus.PREPARED


def test_stable_tag_is_short_deterministic_and_does_not_expose_inputs():
    first = build_qiskit_submission_tag(
        submission_key="secret-campaign-item",
        spec_hash="spec-v1:secret-digest-material",
    )
    second = build_qiskit_submission_tag(
        submission_key="secret-campaign-item",
        spec_hash="spec-v1:secret-digest-material",
    )

    assert first == second
    assert len(first) < 86
    assert "secret-campaign-item" not in first
    assert "secret-digest-material" not in first


def test_submission_tags_are_merged_before_submit_and_bounded():
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    options = with_qiskit_submission_tags(
        {"environment": {"log_level": "INFO", "job_tags": ["existing"]}},
        stable_tag=stable_tag,
        submission_tags=["campaign-safe", "existing"],
    )

    assert options["environment"] == {
        "log_level": "INFO",
        "job_tags": ["existing", "campaign-safe", stable_tag],
    }

    with pytest.raises(ValueError, match="at most"):
        with_qiskit_submission_tags(
            None,
            stable_tag=stable_tag,
            submission_tags=[f"tag-{index}" for index in range(QISKIT_MAX_JOB_TAGS)],
        )

    with pytest.raises(ValueError, match="iterable of strings"):
        with_qiskit_submission_tags(
            None,
            stable_tag=stable_tag,
            submission_tags="not-a-list",
        )


def test_durable_submit_persists_reference_and_restart_uses_journal_id(
    tmp_path: Path,
    monkeypatch,
):
    _patch_sampler(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "qiskit.sqlite"

    first = asyncio.run(_durable_submit(runtime_config=runtime_config, journal_path=journal_path))

    sampler = _Sampler.instances[0]
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    assert stable_tag in sampler.options["environment"]["job_tags"]
    assert sampler.run_calls == [{"pubs": ["pub-0"], "shots": 100}]
    assert first["job_id"] == "runtime-job-1"
    assert first["submission_key"] == "submission-1"
    assert first["spec_hash"] == "spec-v1:abc"
    assert first["journal_path"] == str(journal_path)

    record = QiskitSubmissionJournal(journal_path).get("submission-1")
    assert record is not None
    assert record.status == QiskitSubmissionStatus.SUBMITTED
    assert record.job_id == "runtime-job-1"
    assert record.job_reference == first

    backend_calls = runtime_config.backend_calls
    second = asyncio.run(_durable_submit(runtime_config=runtime_config, journal_path=journal_path))

    assert second == first
    assert len(_Sampler.instances) == 1
    assert runtime_config.backend_calls == backend_calls
    assert runtime_config.service.jobs_calls == []


def test_durable_estimator_sets_tag_before_submit(tmp_path: Path, monkeypatch):
    _patch_estimator(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "estimator.sqlite"

    reference = asyncio.run(
        submit_estimator_job_task.fn(
            ["estimator-pub"],
            runtime_config=runtime_config,
            precision=0.01,
            submission_key="estimator-submission",
            spec_hash="spec-v1:estimator",
            journal_path=journal_path,
        )
    )

    stable_tag = build_qiskit_submission_tag(
        submission_key="estimator-submission",
        spec_hash="spec-v1:estimator",
    )
    estimator = _Estimator.instances[0]
    assert estimator.options["environment"]["job_tags"] == [stable_tag]
    assert estimator.run_calls == [{"pubs": ["estimator-pub"], "precision": 0.01}]
    assert reference["primitive"] == "estimator"
    assert reference["precision"] == 0.01
    assert QiskitSubmissionJournal(journal_path).get("estimator-submission").job_id == (
        "estimator-runtime-job-1"
    )


def test_restart_after_submit_record_crash_attaches_by_tag(
    tmp_path: Path,
    monkeypatch,
):
    class _SimulatedProcessCrash(BaseException):
        pass

    _patch_sampler(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "qiskit.sqlite"
    original_mark_submitted = QiskitSubmissionJournal.mark_submitted
    crashed = False

    def crash_once(self, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise _SimulatedProcessCrash()
        return original_mark_submitted(self, **kwargs)

    monkeypatch.setattr(QiskitSubmissionJournal, "mark_submitted", crash_once)

    with pytest.raises(_SimulatedProcessCrash):
        asyncio.run(_durable_submit(runtime_config=runtime_config, journal_path=journal_path))

    prepared = QiskitSubmissionJournal(journal_path).get("submission-1")
    assert prepared is not None
    assert prepared.status == QiskitSubmissionStatus.PREPARED
    assert prepared.job_id is None
    assert len(runtime_config.service.search_results) == 1

    recovered = asyncio.run(
        _durable_submit(runtime_config=runtime_config, journal_path=journal_path)
    )

    assert recovered["job_id"] == "runtime-job-1"
    assert len(_Sampler.instances) == 1
    assert len(runtime_config.service.jobs_calls) == 1
    search_call = runtime_config.service.jobs_calls[0]
    assert search_call["limit"] is None
    assert search_call["backend_name"] == "ibm_kawasaki"
    assert search_call["job_tags"] == [prepared.stable_tag]
    assert search_call["created_after"] < search_call["created_before"]
    assert QiskitSubmissionJournal(journal_path).get("submission-1").status == (
        QiskitSubmissionStatus.SUBMITTED
    )


def test_zero_candidates_waits_during_grace_then_enters_hold(
    tmp_path: Path,
    monkeypatch,
):
    _patch_sampler(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "qiskit.sqlite"
    journal = QiskitSubmissionJournal(journal_path)
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    journal.prepare(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
        stable_tag=stable_tag,
    )

    with pytest.raises(QiskitRecoveryPending):
        asyncio.run(_durable_submit(runtime_config=runtime_config, journal_path=journal_path))
    assert journal.get("submission-1").status == QiskitSubmissionStatus.PREPARED

    with pytest.raises(QiskitOperatorActionRequired, match="No matching"):
        asyncio.run(
            _durable_submit(
                runtime_config=runtime_config,
                journal_path=journal_path,
                recovery_grace_seconds=0,
            )
        )
    assert journal.get("submission-1").status == QiskitSubmissionStatus.AWAITING_OPERATOR
    assert _Sampler.instances == []


def test_multiple_candidates_enter_hold_without_latest_selection(
    tmp_path: Path,
    monkeypatch,
):
    _patch_sampler(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "qiskit.sqlite"
    journal = QiskitSubmissionJournal(journal_path)
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    journal.prepare(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
        stable_tag=stable_tag,
    )
    runtime_config.service.search_results = [
        _Job("older", backend=runtime_config.backend, tags=[stable_tag]),
        _Job("newer", backend=runtime_config.backend, tags=[stable_tag]),
    ]

    with pytest.raises(QiskitOperatorActionRequired, match="Found 2 matching"):
        asyncio.run(_durable_submit(runtime_config=runtime_config, journal_path=journal_path))

    record = journal.get("submission-1")
    assert record is not None
    assert record.status == QiskitSubmissionStatus.AWAITING_OPERATOR
    assert record.job_id is None
    assert _Sampler.instances == []


def test_candidate_without_stable_tag_enters_hold(tmp_path: Path, monkeypatch):
    _patch_sampler(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "qiskit.sqlite"
    journal = QiskitSubmissionJournal(journal_path)
    stable_tag = build_qiskit_submission_tag(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
    )
    journal.prepare(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
        stable_tag=stable_tag,
    )
    runtime_config.service.search_results = [
        _Job("wrong-tag", backend=runtime_config.backend, tags=["different"]),
    ]

    with pytest.raises(QiskitOperatorActionRequired, match="does not carry"):
        asyncio.run(_durable_submit(runtime_config=runtime_config, journal_path=journal_path))

    record = journal.get("submission-1")
    assert record is not None
    assert record.status == QiskitSubmissionStatus.AWAITING_OPERATOR
    assert record.job_id is None


def test_spec_mismatch_stops_before_submit_or_search(tmp_path: Path, monkeypatch):
    _patch_sampler(monkeypatch)
    runtime_config = _RuntimeConfig()
    journal_path = tmp_path / "qiskit.sqlite"
    journal = QiskitSubmissionJournal(journal_path)
    journal.prepare(
        submission_key="submission-1",
        spec_hash="spec-v1:abc",
        stable_tag=build_qiskit_submission_tag(
            submission_key="submission-1",
            spec_hash="spec-v1:abc",
        ),
    )

    with pytest.raises(QiskitSpecHashMismatchError):
        asyncio.run(
            _durable_submit(
                runtime_config=runtime_config,
                journal_path=journal_path,
                spec_hash="spec-v1:different",
            )
        )

    assert runtime_config.backend_calls == 0
    assert runtime_config.service.jobs_calls == []
    assert _Sampler.instances == []
    assert journal.get("submission-1").spec_hash == "spec-v1:abc"


def test_qiskit_identity_runtime_validates_tags_backend_and_time_window():
    service = _Service()
    backend = _Backend(service)
    now = datetime.now(timezone.utc)
    service.search_results = [
        _Job("valid", backend=backend, tags=["stable"], creation_date=now),
        _Job("wrong-tag", backend=backend, tags=["other"], creation_date=now),
        _Job(
            "future",
            backend=backend,
            tags=["stable"],
            creation_date=now + timedelta(hours=1),
        ),
    ]
    identity = QiskitJobIdentity(
        stable_tag="stable",
        backend_name="ibm_kawasaki",
        search_start=now - timedelta(minutes=1),
        search_end=now + timedelta(minutes=1),
    )

    candidates = QiskitIdentityRecoveryRuntime(service).find_candidates_by_identity(identity)

    assert [(candidate.job_id, candidate.identity_matches) for candidate in candidates] == [
        ("valid", True),
        ("wrong-tag", False),
        ("future", True),
    ]
    assert candidates[0].metadata_error is None
    assert candidates[2].metadata_error == "job future is beyond the recovery clock-skew window"
