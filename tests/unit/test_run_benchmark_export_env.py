"""BenchmarkRunner export env: strict JOB_ID / GCP_PROJECT names; disable callback without crashing."""

from __future__ import annotations

import os

# Import run_benchmark before tests that set this flag would run; keep TF off in unit tests.
os.environ.setdefault("TEMPUSBENCH_DISABLE_TENSORBOARD", "1")

from unittest.mock import MagicMock

import pytest

from tempus_bench.run_benchmark import _resolve_job_id_and_results_callback


def test_callback_none_returns_job_id_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_ID", raising=False)
    jid, cb = _resolve_job_id_and_results_callback(None)
    assert jid is None and cb is None


def test_callback_none_with_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_ID", "  rid  ")
    jid, cb = _resolve_job_id_and_results_callback(None)
    assert jid == "rid" and cb is None


def test_callback_set_missing_job_id_disables_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_ID", raising=False)
    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.setenv("BQ_BUFFER_RESULTS", "1")
    mock_cb = MagicMock()
    jid, cb = _resolve_job_id_and_results_callback(mock_cb)
    assert jid is None and cb is None


def test_callback_set_missing_gcp_disables_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_ID", "j")
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "not-used")
    monkeypatch.setenv("BQ_BUFFER_RESULTS", "1")
    mock_cb = MagicMock()
    jid, cb = _resolve_job_id_and_results_callback(mock_cb)
    assert jid == "j" and cb is None


def test_callback_set_missing_bq_buffer_disables_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_ID", "j")
    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.delenv("BQ_BUFFER_RESULTS", raising=False)
    mock_cb = MagicMock()
    jid, cb = _resolve_job_id_and_results_callback(mock_cb)
    assert jid == "j" and cb is None


def test_callback_set_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_ID", "j")
    monkeypatch.setenv("GCP_PROJECT", "p")
    monkeypatch.setenv("BQ_BUFFER_RESULTS", "1")
    mock_cb = MagicMock()
    jid, cb = _resolve_job_id_and_results_callback(mock_cb)
    assert jid == "j" and cb is mock_cb
