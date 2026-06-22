# Playbook 02 — NCCL Communication Failure

**Symptom class:** a multi-rank job stalls then dies with a NCCL watchdog
**collective timeout** or a `DistBackendError`. One or more ranks never complete a
collective (allreduce/allgather), so peers block until the watchdog aborts.

**Reproduce:** `scripts/run-scenario.sh 02-nccl-failure`

---

## 1. First 60 seconds
```bash
kubectl -n gpu-lab get pods -l job-name=trainer-nccl -o wide
for p in $(kubectl -n gpu-lab get pods -l job-name=trainer-nccl -o name); do
  echo "== $p =="; kubectl -n gpu-lab logs "$p" --tail=15; done
```
Always read **every rank's** logs — the rank that *reports* the timeout is rarely
the rank that *caused* it.

## 2. Hypotheses (ranked)

### H1 — A rank died / never joined, others time out *(most likely)*
- **Evidence:** rank 0 logs `Watchdog caught collective operation timeout:
  WorkNCCL(OpType=ALLREDUCE ...) ran for 600000 ms before timing out`; a peer
  pod is `Error`/`OOMKilled`/missing. Timeout = `WORLD_SIZE` mismatch in practice.
- **Verify:** `kubectl -n gpu-lab get pods -l job-name=trainer-nccl` — count Running
  vs total; inspect the dead rank's `--previous` logs for the *real* root cause
  (often an OOM or assert on one rank).

### H2 — Network/topology: ranks can't reach each other
- **Evidence:** `NCCL WARN NET/IB : No device found` (falls back to slow socket),
  connection-refused/`unable to connect` to `MASTER_ADDR:MASTER_PORT`, or hangs at
  `Bootstrap`. Common with wrong headless Service / DNS / NetworkPolicy.
- **Verify:**
  ```bash
  kubectl -n gpu-lab exec <pod> -- getent hosts trainer-nccl   # DNS resolves?
  kubectl -n gpu-lab exec <pod> -- sh -c 'nc -vz trainer-nccl 29500'
  ```
  Re-run with `NCCL_DEBUG=INFO` (already set) and `NCCL_SOCKET_IFNAME=eth0`.

### H3 — Mismatched collective (different shapes/counts across ranks)
- **Evidence:** `Mismatched collective detected ... count=262144 but rank 0 called
  ... count=131072`, `ncclInvalidUsage`. Caused by data-dependent control flow,
  uneven batch sharding, or conditional layers that differ per rank.
- **Verify:** set `TORCH_DISTRIBUTED_DEBUG=DETAIL` — it logs the op + shapes per
  rank so you can see which rank diverged. (Switch scenario env `FAILURE_MODE=nccl_mismatch`.)

### H4 — Version / NCCL incompatibility, or IB/GDR misconfig
- **Evidence:** `NCCL version 2.x.y` differs across images; `unhandled cuda error`,
  `unhandled system error`. Multi-image rollouts are a classic trigger.
- **Verify:** confirm a single image digest across all ranks; `NCCL_DEBUG=INFO`
  banner shows version + transport (NVLink/PCI/Net) on each rank.

## 3. Expected log (real signature)
```
[rank0] [E] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=42, OpType=ALLREDUCE, Timeout(ms)=600000) ran for 600351
milliseconds before timing out.
[rank2] torch.distributed.DistBackendError: NCCL communicator was aborted on
rank 2.  Original reason: [Rank 2] Timeout ... ProcessGroupNCCL.cpp:874
```

## 4. Expected metrics
- `Tensor Core Active` → 0 across ranks (no compute; everyone blocked in comms).
- `GPU Utilization` spikes to 100% (busy-spin in the collective) **or** flatlines,
  depending on `TORCH_NCCL_BLOCKING_WAIT`.
- No `loss=` lines advancing → job is not making progress.

## 5. Root cause & fix
**Root cause (this scenario):** ranks 2 and 3 never reached the rendezvous, so
rank 0's allreduce watchdog hit its 600s timeout and aborted the group.
**Fix:** (1) find and fix the *non-joining* rank (here: scheduling/startup — ensure
all `parallelism` pods are Running before training starts; use a barrier with a
generous init timeout); (2) make the headless Service + `MASTER_ADDR` correct; (3)
set a sane `init_process_group(timeout=...)` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
so a dead rank fails the job fast instead of hanging 10 minutes; (4) pin one image
digest across all ranks.
**Guardrail:** alert on "job Running but `Tensor Core Active`==0 for >2m".
