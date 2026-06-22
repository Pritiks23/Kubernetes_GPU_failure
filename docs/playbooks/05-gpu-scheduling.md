# Playbook 05 — Failed GPU Scheduling (Pending / Unschedulable)

**Symptom class:** pod stuck `Pending`, never assigned a node. `kubectl describe`
shows a `FailedScheduling` event. With GPUs the usual line is
`Insufficient nvidia.com/gpu`.

**Reproduce:** `scripts/run-scenario.sh 05-gpu-scheduling` (requests 8 GPUs; nodes have 4)

---

## 1. First 60 seconds
```bash
kubectl -n gpu-lab get pod -l app=trainer-bigask          # STATUS=Pending
kubectl -n gpu-lab describe pod <pod> | sed -n '/Events:/,$p'
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'
```

## 2. Hypotheses (ranked)

### H1 — Requesting more GPUs than any single node has *(most likely)*
- **Evidence:** `0/3 nodes are available: 2 Insufficient nvidia.com/gpu, 1 node(s)
  had untolerated taint`. Request=8, node capacity=4. A single pod can only use
  GPUs on **one** node.
- **Verify:** compare `requests.nvidia.com/gpu` to per-node capacity (command above).
  Fix: lower the request, or shard across pods (one GPU per pod, `WORLD_SIZE`>1).

### H2 — GPUs exist but are all allocated (cluster full)
- **Evidence:** capacity 4 but `allocatable`/free is 0 because other pods hold them.
- **Verify:**
  ```bash
  kubectl describe node <gpu-node> | sed -n '/Allocated resources/,/Events/p'
  ```
  Look at `nvidia.com/gpu  Requests`. Fix: free GPUs, add nodes, use quotas/priority.

### H3 — Device plugin not running → GPUs not advertised at all
- **Evidence:** node capacity shows **no** `nvidia.com/gpu` key (or `<none>`).
- **Verify:** in a real cluster, `kubectl -n kube-system get pods | grep nvidia-device-plugin`;
  in this lab, re-run `scripts/simulate-gpu-nodes.sh`. The scheduler treats an
  unadvertised resource as "no node has it" → permanent Pending.

### H4 — Taints / nodeSelector / affinity exclude the GPU nodes
- **Evidence:** `had untolerated taint`, or `node(s) didn't match Pod's node
  affinity/selector`. GPU nodes are often tainted `nvidia.com/gpu=present:NoSchedule`.
- **Verify:** `kubectl describe node <gpu-node> | grep -i taint`; add matching
  `tolerations` / fix `nodeSelector`.

## 3. Expected events
```
Events:
  Type     Reason            Message
  ----     ------            -------
  Warning  FailedScheduling  0/3 nodes are available: 1 node(s) had untolerated
                             taint {node-role.kubernetes.io/control-plane: },
                             2 Insufficient nvidia.com/gpu.
                             preemption: 0/3 nodes are available: 2 No preemption
                             victims found for incoming pod.
```

## 4. Root cause & fix
**Root cause (this scenario):** the pod requests 8 `nvidia.com/gpu` but the largest
node advertises 4, so the scheduler can never satisfy it on a single node.
**Fix:** request ≤ per-node GPU count; for multi-GPU training, run one GPU per pod
and scale with `parallelism`/a `JobSet`/operator so the scheduler can spread ranks
across nodes. Use `ResourceQuota` + `PriorityClass` to manage contention, and a
`gang-scheduler` (Kueue / Volcano) so multi-rank jobs schedule all-or-nothing.
**Guardrail:** alert on pods Pending > 5m with reason `FailedScheduling`.
