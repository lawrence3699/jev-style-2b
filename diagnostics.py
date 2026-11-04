"""Structured backend diagnostics without request content or credentials.

Tracebacks contain frame locations only: never source lines, locals or a raw
exception rendered by logging's exc_info. Known request strings and their words
are redacted, including excerpts copied into exceptions by downstream libraries.
"""
import json
import logging
import math
import re
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

MESSAGE_LIMIT = 16_384
_WORDS = re.compile(r"\w+", re.UNICODE)
_SECRET = re.compile(
    r"\b(?:hf_[A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_\-]{6,}|AKIA[A-Z0-9]{16}|apikey_[a-fA-F0-9]+_[a-fA-F0-9]+)\b"
    r"|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
)
_AUTH = re.compile(r"(?i)\b(?:bearer|basic)\s+[^\s,;\"'<>]+")
_CREDENTIAL = re.compile(
    r"(?i)([\"']?(?:authorization|proxy-authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|secret|token|cookie|set-cookie)[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;<>]+)"
)
_URL = re.compile(r"https?://[^\s<>\"']+")
_COUNT_KEYS = ("state", "heads", "questions", "images", "image_tokens", "overflow")


def configure_logger(name="jevstyle.demo"):
    """Keep app JSON events visible even when root handlers filter INFO."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)


def _safe_str(error):
    try:
        return str(error)
    except Exception:
        return "<exception message unavailable>"


def _safe_url(match):
    try:
        parts = urlsplit(match.group())
        # Userinfo, query and fragment may contain credentials or submitted data.
        host = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, host, parts.path,
                           "[REDACTED]" if parts.query else "", ""))
    except ValueError:
        return "[REDACTED_URL]"


def sanitize(message: str, *, private_values=()) -> str:
    """Keep diagnostic prose, removing known input (including excerpts) and secrets.

    Redacting matching input words is deliberately conservative: a diagnostic
    word also present in the input may be hidden. Quota fields are parsed before
    this step, from narrowly matched numbers, so they remain useful.
    """
    result = _URL.sub(_safe_url, message[:MESSAGE_LIMIT])
    result = _AUTH.sub("[REDACTED_AUTH]", result)
    result = _CREDENTIAL.sub(r"\1[REDACTED]", result)
    result = _SECRET.sub("[REDACTED_SECRET]", result)
    private_words = set()
    private_unicode = set()
    for value in _strings(private_values):
        if not value:
            continue
        private_unicode.update(char for char in value if ord(char) > 127)
        private_words.update(word.casefold() for word in _WORDS.findall(value))
        # Also cover escaped JSON/Python representations of arbitrary user text.
        for variant in {value, repr(value)[1:-1], json.dumps(value, ensure_ascii=True)[1:-1]}:
            private_words.update(word.casefold() for word in _WORDS.findall(variant))
            if len(variant) >= 8:
                result = result.replace(variant, "[REDACTED_INPUT]")
    result = _WORDS.sub(lambda m: "[REDACTED_INPUT]" if m.group().casefold() in private_words else m.group(), result)
    # CJK excerpts do not necessarily align with whitespace-delimited words.
    return "".join("[REDACTED_INPUT]" if char in private_unicode else char for char in result)


def quota_details(message: str) -> dict:
    """Extract only numeric/duration quota facts explicitly present in an error."""
    if not re.search(r"(?i)(?:zerogpu|gpu quota|quota exceeded)", message):
        return {}
    result = {}
    number = r"(\d+(?:\.\d+)?)"
    patterns = {
        "requested_seconds": rf"\brequested\s*[:=]?\s*{number}\s*(?:s(?:ec(?:ond)?s?)?)?\b",
        "remaining_seconds": rf"\b(?:remaining|available|left)\s*[:=]?\s*{number}\s*(?:s(?:ec(?:ond)?s?)?)?\b",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, message, re.I)
        if match:
            result[key] = float(match.group(1))
    match = re.search(rf"\b{number}\s*s(?:ec(?:ond)?s?)?\s+requested\b", message, re.I)
    if match:
        result.setdefault("requested_seconds", float(match.group(1)))
    # Common platform message: "requested 9s on 5s remaining".
    match = re.search(rf"\b{number}\s*s(?:ec(?:ond)?s?)?\s+(?:remaining|left|available)\b", message, re.I)
    if match:
        result.setdefault("remaining_seconds", float(match.group(1)))
    match = re.search(rf"\brequested\s+{number}\s+(?:of|on)\s+{number}\s+seconds\b", message, re.I)
    if match:
        result.setdefault("requested_seconds", float(match.group(1)))
        result.setdefault("remaining_seconds", float(match.group(2)))
    match = re.search(r"(?i)\b(?:try again in|resets? in)\s+(\d{1,4}:[0-5]\d:[0-5]\d)\b", message)
    if match:
        result["reset_after"] = match.group(1)
    match = re.search(r"(?i)\bresets?(?:_at| at)\s*[:=]?\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)", message)
    if match:
        result["resets_at"] = match.group(1)
    return result


def _counts(counts):
    return {key: value for key, value in (counts or {}).items()
            if key in _COUNT_KEYS and isinstance(value, int) and not isinstance(value, bool) and value >= 0}


def _exception_record(error, private_values):
    message = _safe_str(error)
    record = {
        "exception_type": f"{type(error).__module__}.{type(error).__name__}",
        "original_message": sanitize(message, private_values=private_values),
        "message_truncated": len(message) > MESSAGE_LIMIT,
        "traceback": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)[-40:]],
    }
    quota = quota_details(message[:MESSAGE_LIMIT])
    if quota:
        record["quota"] = quota
    return record


def log_failure(logger, event: str, error: BaseException, *, private_values=(), counts=None,
                requested_seconds=None, trace_id=None, input_error=False) -> str:
    """Emit one JSON event and return its correlation ID; never attach exc_info."""
    trace_id = trace_id or uuid.uuid4().hex
    record = {"event": event, "trace_id": trace_id, "input_counts": _counts(counts),
              "requested_seconds": requested_seconds, **_exception_record(error, private_values)}
    chain, seen, current = [], {id(error)}, error
    for _ in range(4):
        cause = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        chain.append(_exception_record(cause, private_values))
        current = cause
    if chain:
        record["causes"] = chain
    logger.log(logging.WARNING if input_error else logging.ERROR, "%s", json.dumps(record, ensure_ascii=True))
    return trace_id


def log_success(logger, *, counts, requested_seconds: int, model_ms: float, trace_id: str) -> None:
    if not isinstance(model_ms, (int, float)) or not math.isfinite(model_ms):
        model_ms = None
    logger.info("%s", json.dumps({"event": "inference_succeeded", "trace_id": trace_id,
                                 "input_counts": _counts(counts), "requested_seconds": requested_seconds,
                                 "model_ms": model_ms}))
