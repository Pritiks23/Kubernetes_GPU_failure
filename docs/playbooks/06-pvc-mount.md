# Playbook 06 — PVC Mount Failure

**Symptom class:** pod stuck `ContainerCreating` (not Pending — it's scheduled but
can't start). Events show `FailedMount` / `FailedAttachVolume` / "claim not found".

**Reproduce:** `scripts/run-scenario.sh 06-pvc-mount` (references a non-existent PVC)

---

## 1. First 60 seconds
```bash
kubectl -n gpu-lab get pod -l app=trainer-pvc          # ContainerCreating
kubectl -n gpu-lab describe pod <pod> | sed -n '/Events:/,$p'
kubectl -n gpu-lab get pvc
```

## 2. Hypotheses (ranked)

### H1 — Referenced PVC doesn't exist *(most likely, this scenario)*
- **Evidence:** `persistentvolumeclaim "nonexistent-dataset" not found`.
- **Verify:** `kubectl -n gpu-lab get pvc` — the claim name in the pod spec isn't
  listed. Fix the `claimName` or create the PVC.

### H2 — PVC exists but is `Pending` (no PV / no provisioner)
- **Evidence:** `kubectl get pvc` shows `Pending`; event
  `waiting for a volume to be created, either by external provisioner ... or
  manually`. Wrong/absent `storageClassName`.
- **Verify:**
  ```bash
  kubectl -n gpu-lab describe pvc <claim> | sed -n '/Events/,$p'
  kubectl get storageclass
  ```

### H3 — RWO volume already mounted on another node (multi-attach)
- **Evidence:** `Multi-Attach error for volume ... Volume is already exclusively
  attached to one node`. Happens on RWO PVCs during reschedule/rolling update.
- **Verify:** find the other pod holding it: `kubectl get pods -A -o wide` + match
  the volume; or use RWX (e.g. NFS) if truly shared.

### H4 — CSI driver / node plugin unhealthy, or mount timeout
- **Evidence:** `Unable to attach or mount volumes ... timed out waiting for the
  condition`, CSI errors in `kubectl describe`.
- **Verify:** check CSI node DaemonSet pods; `dmesg`/kubelet logs on the node for
  mount errors; permissions/`fsGroup` mismatches.

## 3. Expected events
```
Events:
  Type     Reason       Message
  ----     ------       -------
  Warning  FailedMount  MountVolume.SetUp failed for volume "data":
                        persistentvolumeclaim "nonexistent-dataset" not found
  Warning  FailedMount  Unable to attach or mount volumes: unmounted
                        volumes=[data], unattached volumes=[data]:
                        timed out waiting for the condition
```

## 4. Root cause & fix
**Root cause (this scenario):** the pod mounts PVC `nonexistent-dataset`, which was
never created, so the kubelet can't set up the volume and the pod never leaves
`ContainerCreating`.
**Fix:** point `claimName` at an existing bound PVC (the lab ships
`checkpoints`), or create the dataset PVC. For shared read-many data use an RWX
volume; for checkpoints prefer per-rank RWO + object storage sync. Ensure a default
`StorageClass` and a healthy CSI provisioner exist.
**Guardrail:** validating webhook that rejects pods referencing non-existent PVCs;
alert on pods `ContainerCreating` > 3m.
