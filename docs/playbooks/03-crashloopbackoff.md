# Playbook 03 — CrashLoopBackOff

**Symptom class:** pod `RESTARTS` climbing, `STATUS = CrashLoopBackOff`. The
container starts, exits, kubelet restarts it with exponential backoff
(10s→20s→40s…→max 5m). CrashLoopBackOff is a *symptom*; the real signal is the
**exit code + previous logs**.

**Reproduce:** `scripts/run-scenario.sh 03-crashloopbackoff`

---

## 1. First 60 seconds
```bash
kubectl -n gpu-lab get pod -l app=trainer-crashloop
kubectl -n gpu-lab logs deploy/trainer-crashloop --previous   # logs from the crashed instance
kubectl -n gpu-lab get pod <pod> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode} {.status.containerStatuses[0].lastState.terminated.reason}{"\n"}'
```

## 2. Decode the exit code (fast classifier)

| Exit | Meaning | Likely cause |
|---|---|---|
| 0 | clean exit but restartPolicy=Always | Job should be a `Job`, not a `Deployment` |
| 1 | unhandled Python exception | app bug / bad config / missing file |
| 137 | SIGKILL (128+9) | host **OOMKilled** or liveness probe kill |
| 139 | SIGSEGV (128+11) | native crash — CUDA illegal access, bad driver, C-ext |
| 143 | SIGTERM (128+15) | graceful shutdown / preemption |

```bash
# In this scenario exit=139 with a preceding "illegal memory access" log.
```

## 3. Hypotheses (ranked)

### H1 — Application crashes early on every start *(most likely)*
- **Evidence:** `--previous` logs show a traceback or fatal CUDA error within
  seconds; `lastState.terminated.exitCode` is consistent (139 here).
- **Verify:** read `--previous` logs (current logs are often empty — container just
  started). Reproduce locally: `docker run -e FAILURE_MODE=crash gpu-lab/trainer`.

### H2 — Misconfig: missing env/secret/config file, wrong entrypoint
- **Evidence:** exit 1/2, log says `KeyError`, `FileNotFoundError`,
  `exec: not found`, `permission denied`.
- **Verify:** `kubectl -n gpu-lab describe pod <pod>` → check mounted
  ConfigMaps/Secrets exist; `kubectl get cm,secret -n gpu-lab`.

### H3 — Liveness/readiness probe killing a slow-starting container
- **Evidence:** exit 137, events show `Liveness probe failed`, restarts align with
  `initialDelaySeconds`. GPU model load can exceed a tight probe.
- **Verify:** `kubectl describe pod` → Events; relax `initialDelaySeconds` /
  `failureThreshold` or add a `startupProbe`.

### H4 — Host OOMKilled
- **Evidence:** exit 137, `lastState.reason=OOMKilled`. → cross-link Playbook 01.
- **Verify:** raise `resources.limits.memory`; check node memory pressure
  (`kubectl describe node`).

## 4. Expected log / events
```
$ kubectl -n gpu-lab logs <pod> --previous
[..][rank0][ERROR] CUDA error: an illegal memory access was encountered
[..][rank0][ERROR] CUDA kernel errors might be asynchronously reported ...

$ kubectl -n gpu-lab describe pod <pod>
  Last State:  Terminated
    Reason:    Error
    Exit Code: 139
  Back-off restarting failed container
```

## 5. Root cause & fix
**Root cause (this scenario):** the workload hits an illegal memory access and the
process dies with SIGSEGV (139) on every start; the Deployment's `restartPolicy:
Always` turns repeated crashes into CrashLoopBackOff.
**Fix:** fix the crash (here: the CUDA kernel fault — pin a known-good driver/CUDA,
or fix the indexing bug); for *batch* workloads use a `Job` (`restartPolicy: Never`
+ `backoffLimit`) so retries are bounded and observable rather than an infinite loop.
**Guardrail:** alert on `kube_pod_container_status_restarts_total` rate; surface
exit codes in the dashboard.
