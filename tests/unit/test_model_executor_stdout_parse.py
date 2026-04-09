"""Subprocess stdout parsing for model_executor JSON payload."""

from __future__ import annotations

import json

import pytest

from tempus_bench.pipeline.model_executor import (
    _JSON_OUTPUT_SENTINEL,
    parse_subprocess_eval_stdout,
)


def test_parse_prefers_sentinel_line() -> None:
    noise = '[{"oops": "not the payload"}]'
    payload = [{"y_pred": [1.0], "mae": 0.1}]
    stdout = f"hello\n{noise}\n{_JSON_OUTPUT_SENTINEL}\n{json.dumps(payload)}"
    assert parse_subprocess_eval_stdout(stdout) == payload


def test_parse_sentinel_accepts_multiline_json_payload() -> None:
    payload = [{"mae": 0.1}, {"mae": 0.2}]
    pretty = json.dumps(payload, indent=2)
    stdout = f"{_JSON_OUTPUT_SENTINEL}\n{pretty}"
    assert parse_subprocess_eval_stdout(stdout) == payload


def test_parse_legacy_uses_last_parseable_array() -> None:
    """Earlier spurious ``[...]`` lines must not win over the final results line."""
    wrong = json.dumps([0])
    good = [{"mae": 0.2, "y_pred": [3.0]}]
    stdout = f"In Model Executor\n{wrong}\n{json.dumps(good)}"
    assert parse_subprocess_eval_stdout(stdout) == good


def test_parse_empty_stdout_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_subprocess_eval_stdout("   ")


def test_parse_no_json_raises() -> None:
    with pytest.raises(ValueError, match="No evaluation results"):
        parse_subprocess_eval_stdout("no array here\nstill nothing")


def test_parse_legacy_skips_trailing_empty_array_line() -> None:
    """A ``[]`` line after real output must not be chosen when sentinel is absent."""
    good = [{"mae": 0.1}]
    stdout = f"{json.dumps(good)}\n[]\n"
    assert parse_subprocess_eval_stdout(stdout) == good


def test_parse_accepts_legacy_results_sentinel_alias() -> None:
    payload = [{"ok": True}]
    stdout = f"__TEMPUSBENCH_MODEL_EXECUTOR_RESULTS__\n{json.dumps(payload)}\n"
    assert parse_subprocess_eval_stdout(stdout) == payload
