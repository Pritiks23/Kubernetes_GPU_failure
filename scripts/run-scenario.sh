#!/usr/bin/env bash
# Apply a scenario and (for failure modes) point the fake DCGM exporter at the
# matching metric shape so Grafana reflects the incident.
#   usage: scripts/run-scenario.sh <name>
#   names: 00-healthy 01-cuda-oom 02-nccl-failure 03-crashloopbackoff
#          04-image-pull-error 05-gpu-scheduling 06-pvc-mount 07-training-hang
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-}"
[ -z "$NAME" ] && { grep -m1 names "$0"; sed -n '4,7p' "$0"; exit 1; }
MANIFEST="$ROOT/k8s/scenarios/${NAME}.yaml"
[ -f "$MANIFEST" ] || { echo "no such scenario: $NAME"; ls "$ROOT/k8s/scenarios"; exit 1; }

# Map scenario -> exporter metric shape (for the observability story).
case "$NAME" in
  01-cuda-oom)      SC=cuda_oom ;;
  02-nccl-failure)  SC=nccl_timeout ;;
  07-training-hang) SC=slow ;;        # util collapses to ~0
  *)                SC=none ;;
esac
echo "==> setting dcgm-exporter SCENARIO=$SC"
kubectl -n gpu-lab set env ds/dcgm-exporter SCENARIO="$SC" >/dev/null 2>&1 || true

echo "==> applying $NAME"
kubectl apply -f "$MANIFEST"

echo "==> watching pods (Ctrl-C to stop). Then run: scripts/investigate.sh <pod>"
kubectl -n gpu-lab get pods -l scenario="${NAME#0*-}" -w 2>/dev/null \
  || kubectl -n gpu-lab get pods -w
