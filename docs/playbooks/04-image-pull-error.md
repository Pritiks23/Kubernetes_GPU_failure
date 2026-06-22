# Playbook 04 — ImagePullBackOff / ErrImagePull

**Symptom class:** pod stuck `Pending`/`Waiting`, container state reason
`ErrImagePull` then `ImagePullBackOff`. The kubelet cannot fetch the image.

**Reproduce:** `scripts/run-scenario.sh 04-image-pull-error`

---

## 1. First 60 seconds
```bash
kubectl -n gpu-lab get pod -l app=trainer-badimage
kubectl -n gpu-lab describe pod <pod> | sed -n '/Events:/,$p'
```
The **Events** section contains the literal pull error — read it first.

## 2. Hypotheses (ranked)

### H1 — Wrong image name / tag (typo or non-existent tag) *(most likely)*
- **Evidence:** `Failed to pull image "...:v999-does-not-exist": ... not found` or
  `manifest unknown`.
- **Verify:** confirm the tag exists in the registry; for kind, confirm the image
  was `kind load`ed:
  ```bash
  docker exec -it gpu-lab-worker crictl images | grep trainer
  kind load docker-image gpu-lab/trainer:latest --name gpu-lab
  ```

### H2 — Private registry auth missing/expired
- **Evidence:** `pull access denied`, `unauthorized: authentication required`,
  `401`.
- **Verify:** is there an `imagePullSecrets` on the pod/serviceaccount?
  ```bash
  kubectl -n gpu-lab get pod <pod> -o jsonpath='{.spec.imagePullSecrets}{"\n"}'
  kubectl -n gpu-lab get secret
  ```
  Recreate: `kubectl create secret docker-registry regcred --docker-server=... ...`

### H3 — Registry unreachable / rate-limited / DNS
- **Evidence:** `dial tcp: i/o timeout`, `no such host`, Docker Hub
  `toomanyrequests: rate limit exceeded`.
- **Verify:** `kubectl -n gpu-lab exec <any running pod> -- nslookup <registry>`;
  check egress/proxy; mirror through a pull-through cache.

### H4 — `imagePullPolicy` vs availability mismatch
- **Evidence:** policy `Never`/`IfNotPresent` but image isn't on the node (common
  in kind/minikube when you forget to load).
- **Verify:** load the image, or set `Always` with a reachable registry.

## 3. Expected events
```
Events:
  Type     Reason     Message
  ----     ------     -------
  Normal   Scheduled  Successfully assigned gpu-lab/trainer-badimage-xxx to gpu-lab-worker
  Normal   Pulling    Pulling image "gpu-lab/trainer:v999-does-not-exist"
  Warning  Failed     Failed to pull image "...": rpc error: ... not found
  Warning  Failed     Error: ErrImagePull
  Normal   BackOff    Back-off pulling image "gpu-lab/trainer:v999-does-not-exist"
  Warning  Failed     Error: ImagePullBackOff
```

## 4. Root cause & fix
**Root cause (this scenario):** the manifest references tag `v999-does-not-exist`,
which is neither in the kind cache nor pullable.
**Fix:** correct the tag to `latest` (or a real digest) and `kind load` the image;
in production pin by **digest** (`@sha256:...`) to avoid tag drift, attach
`imagePullSecrets` for private registries, and run a registry mirror to dodge
rate limits.
**Guardrail:** admission policy that rejects `:latest` and floating tags; CI check
that the referenced digest exists before rollout.
