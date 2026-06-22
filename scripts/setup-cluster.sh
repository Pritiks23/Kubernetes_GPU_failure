#!/usr/bin/env bash
# Create a local kind cluster, build + load the lab images, advertise fake GPUs,
# and deploy monitoring. Idempotent: safe to re-run.
set -euo pipefail

CLUSTER="${CLUSTER:-gpu-lab}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Checking prerequisites"
for bin in kind kubectl docker; do
  command -v "$bin" >/dev/null 2>&1 || { echo "MISSING: $bin. Install it first (see README)."; exit 1; }
done
docker info >/dev/null 2>&1 || { echo "Docker daemon not running. Start Docker Desktop."; exit 1; }

echo "==> Creating kind cluster '$CLUSTER' (control-plane + 2 workers)"
if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config k8s/kind-cluster.yaml
else
  echo "    cluster already exists, reusing"
fi
kubectl config use-context "kind-${CLUSTER}"

echo "==> Building images"
docker build -f docker/Dockerfile.trainer  -t gpu-lab/trainer:latest  .
docker build -f docker/Dockerfile.exporter -t gpu-lab/exporter:latest .

echo "==> Loading images into kind"
kind load docker-image gpu-lab/trainer:latest  --name "$CLUSTER"
kind load docker-image gpu-lab/exporter:latest --name "$CLUSTER"

echo "==> Creating namespace + base objects"
kubectl apply -f k8s/00-namespace.yaml

echo "==> Advertising fake nvidia.com/gpu resources on worker nodes"
bash scripts/simulate-gpu-nodes.sh

echo "==> Deploying fake DCGM exporter + Prometheus + Grafana"
kubectl apply -f k8s/monitoring/

echo "==> Waiting for monitoring to be Ready"
kubectl -n gpu-lab rollout status deploy/prometheus --timeout=120s || true
kubectl -n gpu-lab rollout status deploy/grafana    --timeout=120s || true

cat <<EOF

==============================================================
 Cluster '$CLUSTER' is ready.
 GPU capacity (simulated):
EOF
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'
cat <<EOF

 Next:
   scripts/run-scenario.sh 01-cuda-oom
   scripts/investigate.sh   <pod-name>

 Grafana:   kubectl -n gpu-lab port-forward svc/grafana 3000:3000   (admin/admin)
 Prometheus:kubectl -n gpu-lab port-forward svc/prometheus 9090:9090
==============================================================
EOF
