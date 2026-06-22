#!/usr/bin/env bash
# One-shot incident triage bundle for a pod -- the commands an on-call platform
# engineer runs in the first 60 seconds. Prints status, events, describe,
# previous+current logs, container exit codes, node GPU allocation, and the
# scheduler's reasoning. Save the output straight into an investigation report.
#   usage: scripts/investigate.sh <pod-name> [namespace]
set -uo pipefail
POD="${1:-}"
NS="${2:-gpu-lab}"
[ -z "$POD" ] && { echo "usage: $0 <pod-name> [namespace]"; kubectl -n "$NS" get pods; exit 1; }

hr(){ printf '\n========== %s ==========\n' "$1"; }

hr "POD STATUS"
kubectl -n "$NS" get pod "$POD" -o wide

hr "PHASE / CONTAINER STATE / EXIT CODES"
kubectl -n "$NS" get pod "$POD" -o jsonpath='
phase={.status.phase}{"\n"}
reason={.status.reason}{"\n"}
{range .status.containerStatuses[*]}container={.name}
  ready={.ready} restarts={.restartCount}
  state={.state}
  lastState={.lastState}
{end}{"\n"}' 2>/dev/null; echo

hr "EVENTS (kubectl describe -> Events)"
kubectl -n "$NS" describe pod "$POD" | sed -n '/Events:/,$p'

hr "SCHEDULING / NODE PLACEMENT"
kubectl -n "$NS" get pod "$POD" -o jsonpath='node={.spec.nodeName}{"\n"}'
echo "-- node GPU allocation across fleet --"
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU_CAP:.status.capacity.'nvidia\.com/gpu',GPU_ALLOC:.status.allocatable.'nvidia\.com/gpu'

hr "CURRENT LOGS (tail 60)"
kubectl -n "$NS" logs "$POD" --tail=60 2>&1 || echo "(no current logs -- container may not have started)"

hr "PREVIOUS LOGS (after a restart/crash; tail 60)"
kubectl -n "$NS" logs "$POD" --previous --tail=60 2>&1 || echo "(no previous container)"

hr "RECENT NAMESPACE EVENTS (sorted)"
kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -25

hr "TRIAGE HINTS"
cat <<'EOF'
  Pending + "Insufficient nvidia.com/gpu"   -> playbook 05 (GPU scheduling)
  Pending + "ErrImagePull/ImagePullBackOff" -> playbook 04 (image pull)
  ContainerCreating + "FailedMount"         -> playbook 06 (PVC mount)
  OOMKilled (137) / CUDA OutOfMemoryError   -> playbook 01 (CUDA OOM)
  CrashLoopBackOff + exit 139/1             -> playbook 03 (crashloop)
  NCCL watchdog timeout / DistBackendError  -> playbook 02 (NCCL)
  Running, 0% GPU util, no log progress     -> playbook 07 (hang)
EOF
