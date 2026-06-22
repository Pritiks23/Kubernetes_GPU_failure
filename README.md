# GPU Platform Failure Investigation Lab

A self-contained lab that reproduces the failure modes of a real
distributed-training **GPU platform on Kubernetes** — CUDA OOM, NCCL timeouts,
CrashLoopBackOff, image-pull errors, GPU scheduling failures, PVC mount failures,
and silent training hangs — with full **Prometheus/Grafana observability** and
**incident-response playbooks + worked root-cause analyses**.

It runs on a laptop with **no GPU and no cloud spend**: GPUs are advertised to the
real Kubernetes scheduler via the built-in node extended-resources API, workloads
run CPU PyTorch but emit CUDA/NCCL-shaped telemetry, and a fake DCGM exporter feeds
real `DCGM_FI_DEV_*` metrics into Prometheus/Grafana. See
[docs/architecture.md](docs/architecture.md) for *why this is honest, not a mock*.

> **Interview framing:** the scheduler decisions, pod lifecycle, events, exit
> codes, PVC binding, scraping, and alerting are all **real Kubernetes behavior**.
> Only the GPU hardware and the exact text of failure logs are simulated.

## What it demonstrates
1. Distributed PyTorch training on Kubernetes (Jobs, indexed multi-rank, headless rendezvous).
2. GPU scheduling concepts — `nvidia.com/gpu` requests/limits, capacity, `Pending`/`FailedScheduling`.
3. Observability — Prometheus scraping a DCGM-style exporter, Grafana GPU dashboard, alert rules.
4. Root-cause analysis — 7 ranked-hypothesis playbooks + 2 full incident postmortems.
5. Incident-response tooling — a one-shot `investigate.sh` triage bundle.

## Repo layout
```
docker/        Dockerfiles (trainer = CPU torch; exporter = fake DCGM)
src/           train.py (failure injector) + gpu_metrics_exporter.py
k8s/           kind config, namespace, monitoring stack, 7 scenarios
scripts/       setup-cluster, simulate-gpu-nodes, run-scenario, investigate, teardown
docs/          architecture + 7 playbooks
logs/          captured example logs per failure
investigations/ worked RCA postmortems (INC-001, INC-002)
```

## Prerequisites
- Docker (Desktop or engine, running)
- [kind](https://kind.sigs.k8s.io/) — `brew install kind`
- kubectl — `brew install kubectl`
- `curl` (used to PATCH node status)

## Quickstart
```bash
cd gpu-platform-failure-lab
scripts/setup-cluster.sh                 # kind up, build+load images, fake GPUs, monitoring
scripts/run-scenario.sh 00-healthy       # green baseline
scripts/run-scenario.sh 01-cuda-oom      # break something
scripts/investigate.sh <pod-name>        # triage bundle -> paste into an RCA
```
Dashboards:
```bash
kubectl -n gpu-lab port-forward svc/grafana 3000:3000      # admin/admin -> "GPU Fleet Overview"
kubectl -n gpu-lab port-forward svc/prometheus 9090:9090   # /alerts
```
Tear down:
```bash
scripts/teardown.sh        # remove scenario workloads only
scripts/teardown.sh --all  # delete the kind cluster
```

## The 7 scenarios
| # | Run | You'll see | Playbook |
|---|---|---|---|
| 00 | `run-scenario.sh 00-healthy` | Job completes, loss drops, util ~90% | — |
| 01 | `run-scenario.sh 01-cuda-oom` | `torch.cuda.OutOfMemoryError`, exit 1, mem→99% | [01](docs/playbooks/01-cuda-oom.md) |
| 02 | `run-scenario.sh 02-nccl-failure` | NCCL watchdog ALLREDUCE timeout, victim-vs-cause | [02](docs/playbooks/02-nccl-failure.md) |
| 03 | `run-scenario.sh 03-crashloopbackoff` | RESTARTS climbing, exit 139, backoff | [03](docs/playbooks/03-crashloopbackoff.md) |
| 04 | `run-scenario.sh 04-image-pull-error` | ErrImagePull → ImagePullBackOff | [04](docs/playbooks/04-image-pull-error.md) |
| 05 | `run-scenario.sh 05-gpu-scheduling` | Pending, `Insufficient nvidia.com/gpu` | [05](docs/playbooks/05-gpu-scheduling.md) |
| 06 | `run-scenario.sh 06-pvc-mount` | ContainerCreating, `FailedMount` | [06](docs/playbooks/06-pvc-mount.md) |
| 07 | `run-scenario.sh 07-training-hang` | Running but 0% util, no progress (deadlock) | [07](docs/playbooks/07-training-hang.md) |

## How a failure is injected
`src/train.py` reads `FAILURE_MODE` and deterministically produces the matching
failure (OOM traceback, NCCL timeout, SIGSEGV, deadlock, straggler). Scheduling,
image, and PVC failures are produced by the **manifests themselves** (over-request
GPUs, bad tag, missing PVC) so the Kubernetes behavior is genuine.

## Worked investigations (read these)
- [INC-001 — CUDA OOM after batch-size bump](investigations/INC-001-cuda-oom.md)
- [INC-002 — NCCL timeout that was really a scheduling/capacity bug](investigations/INC-002-nccl-timeout.md)

## Key findings / talking points
- **Exit code is the fastest classifier:** 1 = CUDA OOM, 137 = host OOMKilled,
  139 = SIGSEGV, 143 = SIGTERM/preemption.
- **In distributed training, read every rank** — the rank that logs the timeout is
  usually the *victim*, not the *cause* (INC-002).
- **A NCCL timeout is often a scheduling problem in disguise** (a rank never started).
- **Hangs need progress-based alerting** — crash-only monitoring never catches a
  deadlock where pods stay `Running` at 0% util.
- **`Pending` vs `ContainerCreating`** splits scheduling failures (05) from
  storage/image failures (04, 06) instantly.

## 1-Day vs Production
See the comparison table in [docs/architecture.md](docs/architecture.md#1-day-vs-production).
Headlines: real GPUs + NVIDIA device plugin, NCCL over NVLink/IB, real
dcgm-exporter, `torchrun` + a gang scheduler (Kueue/Volcano), and
Loki/Alertmanager.



<img width="2842" height="1766" alt="image" src="https://github.com/user-attachments/assets/d2ec1c85-403f-46de-9db7-d0efce932046" />
<img width="1352" height="422" alt="image" src="https://github.com/user-attachments/assets/fa2c03a9-99dd-412d-8854-005326922912" />
<img width="1948" height="730" alt="image" src="https://github.com/user-attachments/assets/11cf8644-dba7-4df5-bf1e-9f1721f2099b" />
<img width="1948" height="1570" alt="image" src="https://github.com/user-attachments/assets/d00e3661-d205-42a1-84a6-320863d3e656" />
<img width="1948" height="1570" alt="image" src="https://github.com/user-attachments/assets/6295ffdf-d463-4cef-a711-5f59a26b5bdf" />
<img width="1948" height="1570" alt="image" src="https://github.com/user-attachments/assets/a8bec440-0e04-48c0-9afe-a69c4a6853df" />
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/4249b82d-46e6-46ff-8c46-54cb406034a5" />
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/781e49d1-30fa-4e86-bbfc-8ec016938731" />
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/a045786e-2524-4a08-9661-d95c9262361b" />
<img width="2880" height="446" alt="image" src="https://github.com/user-attachments/assets/2ae39438-fd4e-4134-bb64-4f8285f6aab5" />

## Findings:
Prometheus is working correctly, and the alerts are definitely firing: you currently have 8 critical GpuMemoryNearOOM alerts and 8 warning GpuIdleButAllocated alerts across both gpu-lab-worker and gpu-lab-worker2, meaning every fake GPU is showing very high memory use (about 99%) while utilization is also very low (0%), which matches an OOM-risk plus stalled-idle pattern rather than normal training. The alerts have stayed active since around 22:48 to 22:49 UTC, so this is not a brief spike, and it explains why you should treat this as an active incident signal in the lab. GpuEccDoubleBitError is 0, which is good and suggests no simulated hardware ECC fault right now.







