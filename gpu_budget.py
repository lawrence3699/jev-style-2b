"""CPU-only ZeroGPU reservation policy; no model or framework imports."""
import math
import threading

MAX_S = 60
SMALL_TEXT_TOKENS = 512
SMALL_TEXT_COLD_S = 12
SMALL_TEXT_WARM_S = 4


def estimate_s(counts: dict, *, warmed_up: bool = False) -> int:
    """Return an uncapped reservation, retaining the old policy for larger work.

    The v4 observations cover only 258-301 token, single-question text calls:
    1.161 s for the first and 0.073/0.072 s subsequently (model time only).
    Reserve 4 s for similar calls after success, and 12 s before the first
    success because earlier v3 notes reported cold workers taking up to 10 s.
    This process-level signal does not guarantee a warm ZeroGPU worker.
    Long text, multiple questions and images retain their conservative budget.
    See docs/zerogpu-budget.md for evidence and deployment checks.
    """
    tokens = counts["state"] + counts["heads"]
    if not counts["images"] and counts["questions"] == 1 and tokens <= SMALL_TEXT_TOKENS:
        return SMALL_TEXT_WARM_S if warmed_up else SMALL_TEXT_COLD_S
    return math.ceil(8 + tokens * 0.0015 + counts["questions"] * 0.5 + counts["images"] * 3)


class DurationPolicy:
    """Remember successful inference in the app process, outside GPU workers."""

    def __init__(self):
        self._warmed_up = False
        self._lock = threading.Lock()

    def estimate(self, counts: dict) -> int:
        with self._lock:
            return estimate_s(counts, warmed_up=self._warmed_up)

    def mark_success(self) -> None:
        with self._lock:
            self._warmed_up = True
