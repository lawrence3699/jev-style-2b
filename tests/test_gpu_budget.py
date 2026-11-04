"""Reservation limits and caller/ZeroGPU agreement, without loading weights."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from diagnostics import log_failure, log_success
from gpu_budget import DurationPolicy, MAX_S, estimate_s


def counts(tokens=291, questions=1, images=0):
    return {"state": tokens - 100, "heads": 100, "questions": questions, "images": images}


@pytest.mark.parametrize("tokens", [258, 291, 301, 304, 512])
def test_small_text_cold_then_warm(tokens):
    policy = DurationPolicy()
    assert policy.estimate(counts(tokens)) == 12
    policy.mark_success()
    assert policy.estimate(counts(tokens)) == 4
    assert DurationPolicy().estimate(counts(tokens)) == 12  # process restart


@pytest.mark.parametrize("tokens,questions,images,expected", [
    (513, 1, 0, 10), (3000, 1, 0, 13), (32768, 1, 0, 58),
    (291, 2, 0, 10), (291, 1, 1, 12), (32768, 1, 1, 61),
    (30000, 12, 0, 59), (31000, 12, 0, 61),
])
def test_unmeasured_workloads_keep_previous_policy(tokens, questions, images, expected):
    c = counts(tokens, questions, images)
    assert estimate_s(c) == estimate_s(c, warmed_up=True) == expected


def app_functions():
    """Load only inference glue; importing app.py would download model weights."""
    tree = ast.parse((Path(__file__).parents[1] / "app.py").read_text())
    names = {"estimate_s", "check_duration", "gpu_seconds", "score_request"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    env = {"DURATION_POLICY": DurationPolicy(), "MAX_S": MAX_S, "uuid": uuid,
           "LOG": logging.getLogger("test.glue"), "log_failure": log_failure, "log_success": log_success}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), env)
    return SimpleNamespace(**env), env


def test_cpu_duration_boundary():
    glue, _ = app_functions()
    glue.check_duration(counts(34333))  # ceil(8 + 51.4995 + .5) == 60
    with pytest.raises(ValueError, match="allows 60 s"):
        glue.check_duration(counts(34334))


def test_gpu_receives_exact_logged_reservation_and_only_success_warms(caplog):
    glue, env = app_functions()
    seen = []

    def score(state, qs, images, requested):
        seen.append(glue.gpu_seconds(state, qs, images, requested))
        if len(seen) == 1:
            raise RuntimeError("ZeroGPU quota exceeded (12s requested vs. 5s left)")
        return [{"answer": "yes"}], 1161.0

    env["score"] = score
    with caplog.at_level(logging.INFO, logger="test.glue"):
        with pytest.raises(RuntimeError):
            glue.score_request("private request", [], [], counts())
        assert env["DURATION_POLICY"].estimate(counts()) == 12
        glue.score_request("private request", [], [], counts())
        glue.score_request("private request", [], [], counts())
    assert seen == [12, 12, 4]
    events = [json.loads(record.getMessage()) for record in caplog.records]
    assert [event["requested_seconds"] for event in events] == seen
    assert events[0]["quota"] == {"requested_seconds": 12.0, "remaining_seconds": 5.0}
    assert events[1]["model_ms"] == 1161.0
    assert all(len(event["trace_id"]) == 32 for event in events)


def test_warm_state_thread_safe():
    policy = DurationPolicy()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: policy.mark_success(), range(100)))
        values = list(pool.map(lambda _: policy.estimate(counts()), range(100)))
    assert set(values) == {4}
