#!/usr/bin/env bash
# Remove scenario workloads (keep cluster + monitoring), or delete everything.
#   scripts/teardown.sh           # delete only scenario workloads
#   scripts/teardown.sh --all     # delete the whole kind cluster
set -euo pipefail
CLUSTER="${CLUSTER:-gpu-lab}"
if [ "${1:-}" = "--all" ]; then
  echo "==> deleting kind cluster '$CLUSTER'"
  kind delete cluster --name "$CLUSTER"
  exit 0
fi
echo "==> deleting scenario workloads in namespace gpu-lab"
kubectl -n gpu-lab delete jobs,deployments,svc -l 'scenario' --ignore-not-found
kubectl -n gpu-lab delete deploy trainer-crashloop trainer-badimage trainer-bigask trainer-pvc --ignore-not-found
kubectl -n gpu-lab delete job trainer-healthy trainer-oom trainer-nccl trainer-hang --ignore-not-found
echo "==> resetting dcgm exporter scenario"
kubectl -n gpu-lab set env ds/dcgm-exporter SCENARIO=none >/dev/null 2>&1 || true
echo "done."
