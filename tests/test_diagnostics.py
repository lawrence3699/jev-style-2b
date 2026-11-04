import json
import logging

import pytest

from diagnostics import MESSAGE_LIMIT, configure_logger, log_failure, log_success, quota_details, sanitize
from recovery import classify_error


@pytest.mark.parametrize("message,expected", [
    ("ZeroGPU quota exceeded (60s requested vs. 30s left). You can try again in 0:01:02",
     {"requested_seconds": 60.0, "remaining_seconds": 30.0, "reset_after": "0:01:02"}),
    ("ZeroGPU quota exceeded: requested 9s on 5.25s remaining; resets_at=2026-11-04T12:00:00Z",
     {"requested_seconds": 9.0, "remaining_seconds": 5.25, "resets_at": "2026-11-04T12:00:00Z"}),
    ("ZeroGPU Quota Exceeded! Requested 9 of 0 seconds.",
     {"requested_seconds": 9.0, "remaining_seconds": 0.0}),
    ("GPU quota exhausted. requested=4 remaining=3 reset in 23:59:59",
     {"requested_seconds": 4.0, "remaining_seconds": 3.0, "reset_after": "23:59:59"}),
    ("ZeroGPU quota exceeded", {}),
    ("Invalid user text: requested 9 remaining 5", {}),
])
def test_quota_facts_only_when_explicit(message, expected):
    assert quota_details(message) == expected


@pytest.mark.parametrize("secret,message", [
    ("hf_FAKEPRIVATE1234567890", "credential hf_FAKEPRIVATE1234567890 rejected"),
    ("sk-proj-FAKEPRIVATE123456", "client sk-proj-FAKEPRIVATE123456 failed"),
    ("apikey_012345abcdef_abcdef012345", "invalid apikey_012345abcdef_abcdef012345"),
    ("arbitrary.private.value", "Authorization: Bearer arbitrary.private.value"),
    ("base64testsecret", "Basic base64testsecret"),
    ("anyRawApiKey", '{"api_key": "anyRawApiKey"}'),
    ("privateCookieValue", "cookie=privateCookieValue"),
    ("privateQuerySecret", "https://example.test/?token=privateQuerySecret"),
    ("user:privatePassword", "https://user:privatePassword@example.test/path"),
])
def test_credentials_are_redacted_without_known_request(secret, message):
    assert secret not in sanitize(message)


def test_input_whole_fragments_unicode_and_escaped_text_removed():
    private = {"text": "Customer Otter42 requests refund. 这是保密文本\nsecond line",
               "question": "Ship the package?", "options": {"yes": "Approve", "no": "Deny"},
               "image": "/tmp/gradio/private-upload.png"}
    message = ('Unknown state Customer Otter42; question "Ship the package?"; '
               'image /tmp/gradio/private-upload.png; escaped \\u8fd9\\u662f\\u4fdd\\u5bc6\\u6587\\u672c\\nsecond line')
    result = sanitize(message, private_values=(private,))
    for fragment in ("Customer", "Otter42", "Ship", "package", "private-upload", "\\u8fd9", "second", "line"):
        assert fragment not in result
    assert "保密" not in sanitize("bad text fragment: 保密", private_values=(private,))


def test_structured_original_exception_chain_and_safe_traceback(caplog):
    logger = logging.getLogger("test.diagnostics")
    private_text = "Customer Otter42 requests refund"
    try:
        try:
            secret_local = "must never be logged"
            raise RuntimeError("ZeroGPU quota exceeded (12s requested vs. 5s left). Try again in 0:01:00")
        except RuntimeError as cause:
            raise ValueError(f"Runtime failed for {private_text}; hf_FAKEPRIVATE1234567890") from cause
    except ValueError as error:
        with caplog.at_level(logging.ERROR, logger=logger.name):
            trace_id = log_failure(logger, "inference_failed", error, private_values=(private_text,),
                                   counts={"state": 105, "heads": 186, "questions": 1, "text": private_text},
                                   requested_seconds=12)
        friendly = classify_error(error).as_markdown()
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.exc_info is None
    raw = record.getMessage()
    event = json.loads(raw)
    assert event["event"] == "inference_failed"
    assert event["trace_id"] == trace_id
    assert event["exception_type"] == "builtins.ValueError"
    assert "Runtime failed for" in event["original_message"]
    assert event["requested_seconds"] == 12
    assert event["input_counts"] == {"state": 105, "heads": 186, "questions": 1}
    assert event["causes"][0]["quota"] == {
        "requested_seconds": 12.0, "remaining_seconds": 5.0, "reset_after": "0:01:00"}
    assert set(event["traceback"][0]) == {"file", "line", "function"}
    for value in (private_text, "Otter42", "hf_FAKEPRIVATE1234567890", secret_local, "raise ValueError"):
        assert value not in raw
        assert value not in friendly


def test_message_bound_and_unprintable_exception(caplog):
    class BrokenError(Exception):
        def __str__(self):
            raise RuntimeError("cannot render")

    with caplog.at_level(logging.ERROR):
        log_failure(logging.getLogger("test.bound"), "failed", RuntimeError("x" * (MESSAGE_LIMIT + 5)))
        log_failure(logging.getLogger("test.bound"), "failed", BrokenError())
    event, broken = [json.loads(record.getMessage()) for record in caplog.records]
    assert event["message_truncated"]
    assert len(event["original_message"]) == MESSAGE_LIMIT
    assert broken["original_message"] == "<exception message unavailable>"


def test_success_info_visible_once_independent_of_root_handlers(capsys):
    logger = configure_logger("test.dedicated_json")
    assert configure_logger("test.dedicated_json") is logger
    try:
        log_success(logger, counts={"state": 105, "heads": 186}, requested_seconds=4,
                    model_ms=72.0, trace_id="a" * 32)
        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["requested_seconds"] == 4
        assert event["model_ms"] == 72.0
        assert event["event"] == "inference_succeeded"
        assert logger.propagate is False
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
