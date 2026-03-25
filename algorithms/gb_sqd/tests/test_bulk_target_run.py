from __future__ import annotations

from gb_sqd.artifact_keys import bulk_metrics_artifact_key


def test_bulk_metrics_artifact_key_normalizes_mode_name():
    assert bulk_metrics_artifact_key("ext_sqd") == "gb-sqd-bulk-ext-sqd-metrics"
    assert bulk_metrics_artifact_key("trim_sqd") == "gb-sqd-bulk-trim-sqd-metrics"


def test_bulk_metrics_artifact_key_falls_back_for_empty_mode():
    assert bulk_metrics_artifact_key("___") == "gb-sqd-bulk-unknown-metrics"
