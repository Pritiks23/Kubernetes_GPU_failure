# INC-001 — Training job CrashLoopBackOff after batch-size bump

| | |
|---|---|
| **Severity** | SEV-3 (single job, no fleet impact) |
| **Status** | Resolved |
| **Date** | 2026-06-22 |
| **Author** | Platform on-call |
| **Affected** | `gpu-lab/trainer-oom` (1 GPU, A100-40GB) |
| **Playbook** | [01 — CUDA OOM](../docs/playbooks/01-cuda-oom.md) |

## Summary
A training Job began failing immediately after a config change increased
`BATCH_SIZE` from 256 to 1024. The container died with
`torch.cuda.OutOfMemoryError` on the first backward pass and the Job exhausted its
`backoffLimit`. **Root cause:** activation memory for the larger batch exceeded the
40 GiB device. No infrastructure fault.

## Impact
- 1 training Job failed to start; 0 successful steps.
- ~4 minutes of GPU idle (3 retry attempts) before the Job was marked failed.
- No other tenants affected (GPU is exclusively scheduled).

## Detection
`GpuMemoryNearOOM` Prometheus alert fired (memory util >95% for 10s), 8s before the
crash. On-call also saw the Job in `BackoffLimitExceeded`.

## Timeline (UTC)
| Time | Event |
|---|---|
| 15:10:02 | Pod starts; CUDA banner shows A100-40GB. |
| 15:10:03 | Training loop begins; `gpu_mem` already 28.7 GiB at batch=1024. |
| 15:10:05 | `GpuMemoryNearOOM` alert fires (mem util ramps to 99%). |
| 15:10:05 | `torch.cuda.OutOfMemoryError`: tried to allocate 2.00 GiB, 312 MiB free. exit 1. |
| 15:10:20–15:10:55 | 2 further retries, each OOMs identically. |
| 15:10:55 | Job → `BackoffLimitExceeded`. Paged. |

## Investigation
Ran `scripts/investigate.sh trainer-oom-abc12`.

1. **Classify OOM type.** `lastState.terminated.reason=Error, exitCode=1` and the
   log tail is a `torch.cuda.OutOfMemoryError` → **CUDA** OOM, *not* host
   `OOMKilled` (which would be exit 137). This rules out the pod memory limit.
2. **H1 (batch too large) — confirmed.** OOM occurs at low `global_step` on the
   first backward; `DCGM_FI_DEV_FB_USED` ramps straight to 100% with no sawtooth.
   The only recent change was `BATCH_SIZE` 256→1024 (4×).
3. **H3 (leak) — rejected.** Memory does not grow step-over-step; it's high from
   step 0. Not a leak.
4. **H4 (co-tenant) — rejected.** "Process 1" in the error is our own PID; GPU is
   exclusively scheduled (`nvidia.com/gpu: 1` limit).

## Root cause
Activation memory scales with batch size. At batch=1024 the forward activations +
gradients + optimizer state exceeded 40 GiB on the first backward, raising
`OutOfMemoryError`. The Deployment-style retries simply repeated the same doomed
allocation.

## Resolution
1. Reverted per-device `BATCH_SIZE` to 256.
2. Restored the intended global batch via **gradient accumulation** (`accum=4`).
3. Enabled **bf16** autocast — halved activation memory, headroom now ~55%.

## Follow-ups / prevention
- [ ] Add a pre-flight memory estimate to CI that rejects configs > device memory.
- [ ] Enable activation checkpointing on the transformer blocks (defense in depth).
- [x] `GpuMemoryNearOOM` alert already in place — fired correctly, keep it.
- [ ] Document batch-size guidance per GPU SKU in the team runbook.

## Lessons
- Exit code is the fastest classifier: **1 = CUDA OOM, 137 = host OOMKilled.** They
  have completely different fix domains.
- A larger global batch should come from accumulation, not a bigger per-device
  batch, when memory-bound.
