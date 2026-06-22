#!/usr/bin/env bash
# Advertise FAKE GPUs on the kind worker nodes using Kubernetes' built-in
# *node-level extended resources* feature -- a real, documented API. We PATCH the
# node /status subresource to add `nvidia.com/gpu` capacity. The scheduler then
# tracks GPU allocation EXACTLY like a real cluster (requests, limits, Pending on
# exhaustion). No device plugin, no real GPU, no driver required.
#
# We also label nodes so workloads can nodeSelect onto "gpu" nodes, mirroring how
# real clusters use node labels like nvidia.com/gpu.product.
set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GPU_PRODUCT="${GPU_PRODUCT:-NVIDIA-A100-SXM4-40GB}"

# kubectl proxy lets us PATCH the status subresource with a JSON merge patch.
echo "    starting kubectl proxy on :8001"
kubectl proxy --port=8001 >/tmp/kubectl-proxy.log 2>&1 &
PROXY_PID=$!
trap 'kill $PROXY_PID >/dev/null 2>&1 || true' EXIT
sleep 2

# Only worker nodes get GPUs (control-plane stays GPU-free, like production).
WORKERS=$(kubectl get nodes -l '!node-role.kubernetes.io/control-plane' \
            -o jsonpath='{.items[*].metadata.name}')

for node in $WORKERS; do
  echo "    advertising ${GPUS_PER_NODE}x nvidia.com/gpu on node '$node'"
  curl -sf --header "Content-Type: application/json-patch+json" \
    --request PATCH \
    --data "[{\"op\":\"add\",\"path\":\"/status/capacity/nvidia.com~1gpu\",\"value\":\"${GPUS_PER_NODE}\"}]" \
    "http://localhost:8001/api/v1/nodes/${node}/status" >/dev/null
  kubectl label node "$node" \
    nvidia.com/gpu.present=true \
    nvidia.com/gpu.product="$GPU_PRODUCT" \
    --overwrite >/dev/null
done

echo "    done. current GPU capacity:"
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu',PRODUCT:.metadata.labels.'nvidia\.com/gpu\.product'
