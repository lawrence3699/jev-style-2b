# ZeroGPU reservation and backend diagnostics

The Hugging Face [ZeroGPU duration documentation](https://huggingface.co/docs/hub/spaces-zerogpu#duration-management)
defines `duration` as the maximum function runtime, supports a callable with the
same arguments as the GPU function, and recommends shorter reservations for
quicker work. Reservations should leave headroom; they are not measured runtime.

The v4 diagnostic run on 2026-11-04 used Space revision
`90fc5d7c301d91bd599c1b5fb520cb559855381f`. Three successful single-question,
text-only calls had 291, 258 and 301 total tokens and reported model runtimes of
1,161, 73 and 72 ms. Their HTTP/SSE elapsed times were 4.9383, 1.7350 and 1.7629 s,
which include queue/network work outside the GPU function. Each requested 9 s
under the old `ceil(8 + .0015*tokens + .5*questions + 3*images)` formula. A fourth
304-token request did not return a live result, so it is not runtime evidence.

For a single question with no image and at most 512 state-plus-question tokens,
the app now requests **12 s before its first successful inference**, then **4 s**.
The cold allowance retains headroom over the earlier v3 source comment reporting
up to 10 s on a cold worker. The warm allowance is more than three times the
largest observed v4 model time. The 512-token boundary is a deliberately narrow
deployment scope, not a measured throughput limit. All other workloads retain
the old formula and the 60 s CPU rejection threshold. The cold tier can exceed
the adjacent larger-text tier; this reflects a conservative cold allowance.

Warm-up status belongs to the app process, is protected by a lock, changes only
after successful inference returns from `@GPU`, and resets with the process.
It does **not** guarantee that a later ZeroGPU worker is warm. No automatic retry
is added. A production check should exercise the first small request and repeated
small requests, then examine actual timeout events before tightening the budget.

`inference_succeeded` JSON events contain a trace ID, requested seconds, numeric
input counts and measured model milliseconds. These milliseconds exclude worker
allocation, queue and network overhead and should not be described as quota use.
`inference_failed` events retain the exception type, sanitized original message,
numeric quota facts when present, and frame locations without source or locals.
Known request strings/words and common credential formats are redacted; this can
also redact diagnostic words shared with the input. Friendly browser recovery
messages remain fixed text. Failures rejected by the platform before a callback
starts cannot be logged by application code.

Local checks (no model downloads or GPU calls):

```sh
python3 -m pytest -q
PYTHONPYCACHEPREFIX=/private/tmp/jev_zerogpu_pycache python3 -m py_compile app.py gpu_budget.py diagnostics.py recovery.py
```
