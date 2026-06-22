# INC-002 — 4-rank job hangs 10 minutes then dies with NCCL collective timeout

| | |
|---|---|
| **Severity** | SEV-2 (multi-node job, GPUs idle 10m) |
| **Status** | Resolved |
| **Date** | 2026-06-22 |
| **Author** | Platform on-call |
| **Affected** | `gpu-lab/trainer-nccl` (4 ranks, 4× A100) |
| **Playbook** | [02 — NCCL Failure](../docs/playbooks/02-nccl-failure.md), [05 — GPU Scheduling](../docs/playbooks/05-gpu-scheduling.md) |

## Summary
A 4-rank distributed job appeared to "hang" for 10 minutes and then failed with a
NCCL `ALLREDUCE` watchdog timeout on rank 0. The reported error was on rank 0, but
the **root cause was on ranks 2 and 3**, which never started: their pods were
`Pending` due to GPU exhaustion on the target node, so they never joined the
rendezvous and rank 0 timed out waiting.

This is the classic **"victim vs. cause"** distributed-training incident: the rank
that logs the error is not the rank that caused it.

## Impact
- 1 multi-rank job, 4 GPUs reserved and idle for ~10 minutes (the NCCL timeout).
- Wasted ~0.67 GPU-hours and delayed the run.

## Detection
`GpuIdleButAllocated` alert fired (`Tensor Core Active`==0 on the 2 *running* ranks
while pods were Running). On-call noticed loss lines never advanced.

## Timeline (UTC)
| Time | Event |
|---|---|
| 15:22:10 | Ranks 0 & 1 pods Running, init NCCL bootstrap, wait at rendezvous. |
| 15:22:10 | Ranks 2 & 3 pods **Pending** — `Insufficient nvidia.com/gpu`. |
| 15:22:40 | `GpuIdleButAllocated` fires (running ranks at 0% tensor activity). |
| 15:32:11 | Rank 0 NCCL watchdog hits 600s timeout on ALLREDUCE → aborts. exit 1. |
| 15:32:11 | Ranks running surface `DistBackendError: communicator aborted`. |
| 15:32:30 | Paged; began triage. |

## Investigation
1. **Read every rank, not just the one that errored.**
   ```bash
   kubectl -n gpu-lab get pods -l job-name=trainer-nccl -o wide
   # 0,1 Running ; 2,3 PENDING  <-- the tell
   ```
   Rank 0's log says `Timeout waiting for ranks [2,3] to join`. So the question
   becomes *why didn't 2 and 3 start?* — not "why did rank 0 time out".
2. **H1 (a rank never joined) — confirmed.** Ranks 2 & 3 were `Pending`.
   `describe` on rank 2 showed `FailedScheduling … Insufficient nvidia.com/gpu`.
3. **Cross to Playbook 05.** Fleet had only 4 free GPUs but other jobs held some;
   the 4-GPU gang couldn't all be placed → 2 ranks stuck Pending.
4. **H2 (network) — rejected.** Running ranks resolved `MASTER_ADDR` and bootstrapped
   fine; the issue was *absent* peers, not unreachable peers.
5. **H3 (mismatched collective) — rejected.** No `ncclInvalidUsage`; counts matched.

## Root cause
The job was launched as 4 independent pods with no **gang scheduling**. The cluster
had insufficient free GPUs to place all 4 ranks; 2 scheduled, 2 stayed Pending.
The 2 running ranks blocked in the first allreduce and rank 0's NCCL watchdog
aborted after the default 600s timeout. The 10-minute "hang" was the timeout window.

## Resolution
1. Freed GPUs / waited for capacity, re-ran — job completed.
2. Switched the job to **gang (all-or-nothing) scheduling** (Kueue/Volcano) so it
   only starts when all 4 GPUs are simultaneously available — no partial starts.
3. Set `init_process_group(timeout=120s)` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so
   a missing rank fails in ~2 min instead of wasting 10.

## Follow-ups / prevention
- [ ] Adopt gang scheduler for all multi-rank jobs (eliminates partial starts).
- [ ] Alert: "multi-rank job with any rank Pending > 2m" → page before the timeout.
- [x] `GpuIdleButAllocated` alert validated (fired correctly).
- [ ] Lower default c10d/NCCL timeouts so silent waits become fast failures.

## Lessons
- **In distributed training, read every rank's logs.** The error surfaces on the
  rank that was *waiting*, not the rank that *failed*.
- A NCCL collective timeout is frequently a **scheduling/capacity** problem wearing
  a networking costume.
- Without gang scheduling, a partially-placed job burns GPUs idling until a timeout.
