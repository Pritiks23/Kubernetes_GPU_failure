# Playbook 01 — CUDA Out Of Memory

**Symptom class:** training pod exits non-zero shortly after starting; logs end in a
`torch.cuda.OutOfMemoryError`. In k8s the Job shows `Failed`; with restarts it
becomes `CrashLoopBackOff`. If the OOM is at the *host* level (not CUDA), you'd
see `OOMKilled` / exit 137 instead — distinguish the two early.

**Reproduce:** `scripts/run-scenario.sh 01-cuda-oom`

---

## 1. First 60 seconds

```bash
kubectl -n gpu-lab get pods -l scenario=cuda-oom
kubectl -n gpu-lab logs job/trainer-oom --tail=40
scripts/investigate.sh <pod>
```

## 2. Triage decision: CUDA OOM vs. host OOMKilled

| Signal | CUDA OOM (GPU memory) | Host OOMKilled (RAM) |
|---|---|---|
| Exit code | 1 (Python exception) | 137 (SIGKILL) |
| Log tail | `torch.cuda.OutOfMemoryError` | (none — process killed) |
| `lastState.terminated.reason` | `Error` | `OOMKilled` |
| Fix domain | model/batch/activations | pod `resources.limits.memory` |

```bash
kubectl -n gpu-lab get pod <pod> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason} {.status.containerStatuses[0].lastState.terminated.exitCode}{"\n"}'
```

## 3. Hypotheses (ranked)

### H1 — Batch size / sequence length too large *(most likely)*
GPU memory ≈ params + optimizer state + **activations**, and activations scale
with batch × seq-len. A config bump is the #1 cause.
- **Evidence for:** OOM happens on the *first* backward pass (`global_step` low);
  `DCGM_FI_DEV_FB_USED` climbs to ~100% right before the crash.
- **Verify:** halve `BATCH_SIZE` and rerun:
  ```bash
  kubectl -n gpu-lab set env job/trainer-oom BATCH_SIZE=256   # was 1024
  ```
  (For a Job you must delete+recreate; edit the manifest and re-apply.)

### H2 — No gradient checkpointing / fragmentation
Reserved-but-unallocated memory is large; the allocator can't find a contiguous block.
- **Evidence:** error text mentions "reserved by PyTorch but unallocated" and
  suggests `expandable_segments:True`.
- **Verify:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and enable
  `torch.utils.checkpoint` on transformer blocks.

### H3 — Memory leak across steps (retained graph / list of tensors)
- **Evidence:** OOM occurs after *many* steps, memory rises monotonically each step.
- **Verify:** plot `DCGM_FI_DEV_FB_USED` over time — a sawtooth that trends up = leak;
  flat-then-spike = H1. Check for `loss.append(loss)` (tensor, not `.item()`) or
  missing `optimizer.zero_grad()`.

### H4 — Co-tenancy: another process is using the GPU
- **Evidence:** "Process N has X GiB in use" where N isn't your PID; happens only on
  some nodes.
- **Verify:** `nvidia-smi` on the node; in k8s, GPUs should be exclusively
  scheduled — if not, MPS/time-slicing is enabled. Check device-plugin config.

## 4. Expected log (real signature)

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 40.00 GiB of which 312.00 MiB is free.
Process 1 has 39.40 GiB memory in use. Of the allocated memory 39.50 GiB is
allocated by PyTorch, and 268.00 MiB is reserved but unallocated. If reserved
but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True to avoid fragmentation.
```

## 5. Expected metrics (Grafana → GPU Fleet Overview)
- `Memory Utilization %` panel ramps to ≥95% then drops to 0 (process died).
- `GpuMemoryNearOOM` Prometheus alert fires `for: 10s` before the crash.

## 6. Root cause & fix
**Root cause:** effective batch exceeded device memory; first backward allocated
activations that didn't fit.
**Fix (in priority order):** (1) reduce per-device batch + use gradient
accumulation to keep global batch; (2) enable activation checkpointing;
(3) mixed precision (bf16/fp16) to halve activation memory; (4) `expandable_segments`
for fragmentation; (5) shard optimizer state (ZeRO/FSDP) for large models.
**Guardrail:** add the `GpuMemoryNearOOM` alert + a pre-flight memory estimate in CI.
