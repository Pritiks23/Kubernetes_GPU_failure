# Playbook 07 — Distributed Training Hang (no crash)

**Symptom class:** the hardest one. Pods stay `Running`, no errors, no restarts —
but **no training progress**: loss/step lines stop advancing and GPU utilization
collapses to ~0%. Nothing "fails", so alerts that watch for crashes miss it.

**Reproduce:** `scripts/run-scenario.sh 07-training-hang`

---

## 1. First 60 seconds
```bash
kubectl -n gpu-lab get pods -l job-name=trainer-hang        # all Running, 0 restarts
for p in $(kubectl -n gpu-lab get pods -l job-name=trainer-hang -o name); do
  echo "== $p =="; kubectl -n gpu-lab logs "$p" --tail=5; done
```
Key tell: timestamps in the logs **stop advancing**. Cross-check Grafana:
`Tensor Core Active`→0, `GPU Utilization`→~0.

## 2. Hypotheses (ranked)

### H1 — Collective deadlock: one rank stopped calling collectives *(most likely)*
A rank entered a branch the others didn't (e.g. only rank 0 checkpoints/evals/logs
to disk synchronously) → others block forever in the next allreduce.
- **Evidence:** peer logs end at `blocked in allreduce waiting on rank 0`; rank 0's
  last line is some rank-0-only operation (checkpoint/eval). Util=0 everywhere.
- **Verify:** dump Python stacks of every rank to see who's *not* in a collective:
  ```bash
  kubectl -n gpu-lab exec <pod> -- sh -c 'kill -SIGABRT 1'   # if faulthandler/py-spy
  # or, better, run with: PYTHONFAULTHANDLER=1, and py-spy dump --pid 1
  ```
  The rank whose stack is **not** in `all_reduce` is the culprit.

### H2 — Data loader stall / deadlock (no batches produced)
- **Evidence:** util 0 on *all* ranks from the start of an epoch; `num_workers` +
  fork + CUDA init issues, or a stuck NFS/`PVC` read. CPU also idle.
- **Verify:** check IO wait on the node; set `num_workers=0` to test; look for
  `DataLoader worker (pid ...) is killed by signal`.

### H3 — NCCL waiting on a slow/unreachable peer (network), not a true deadlock
- **Evidence:** with `NCCL_DEBUG=INFO`, repeated `still waiting ... in ALLREDUCE`;
  one node has degraded NIC / different subnet. → overlaps Playbook 02.
- **Verify:** `TORCH_NCCL_BLOCKING_WAIT=1` + a finite `init_process_group(timeout=)`
  converts the silent hang into a loud timeout you can catch.

### H4 — Host-level: GPU fell off the bus / Xid error / ECC
- **Evidence:** `nvidia-smi` shows `ERR!` or a missing GPU; `dmesg` has `Xid` /
  `GPU has fallen off the bus`. DCGM `ECC_DBE` counter increments.
- **Verify:** `dmesg -T | grep -i xid`; DCGM dashboard ECC panel; drain/cordon the
  node and replace.

## 3. Expected log / metrics
```
[rank1][WARN] blocked in c10d allreduce waiting on rank 0...
[rank1][WARN] NCCL INFO still waiting for rank 0 in ALLREDUCE (no progress)
# rank 0:
[rank0][INFO] entering long synchronous checkpoint (simulated stall)...
[rank0][INFO] still inside checkpoint... (rank 0 not calling allreduce)
```
Grafana: `Tensor Core Active`==0 and `GPU Utilization`≈0 while pods Running →
`GpuIdleButAllocated` alert fires.

## 4. Root cause & fix
**Root cause (this scenario):** rank 0 performs a long synchronous operation and
stops participating in collectives; all other ranks block in allreduce → global
deadlock with no crash.
**Fix:** never put rank-divergent blocking work between collectives — checkpoint
**asynchronously** or wrap rank-0-only work so all ranks still hit the same
collective schedule; set finite NCCL/c10d timeouts +
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so a stall becomes a fail-fast error; add a
heartbeat/`activeDeadlineSeconds` (this Job has one) so hung jobs self-terminate.
**Guardrail (most important):** progress-based alerting — alert when a Running
GPU job has `Tensor Core Active`==0 or no log-progress for N minutes. Crash-only
monitoring will never catch a hang.
