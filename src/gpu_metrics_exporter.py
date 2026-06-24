#!/usr/bin/env python3
"""
Fake DCGM-style GPU metrics exporter.

Real clusters run nvidia/dcgm-exporter as a DaemonSet to expose per-GPU
Prometheus metrics. We have no physical GPU, so this stand-in emits the SAME
metric names (DCGM_FI_DEV_*) with realistic, correlated values. Prometheus
scrapes it, Grafana dashboards render it, and the failure scenarios move the
needle (util drops, mem spikes to OOM, ECC errors appear) so the observability
story is end-to-end.

Listens on :9400/metrics (the dcgm-exporter default port).

Env:
  GPU_COUNT       number of fake GPUs on this node (default 1)
  SCENARIO        none|cuda_oom|slow|nccl_timeout  -> shapes the metric values
"""
import http.server
import math
import os
import socketserver
import time

PORT = int(os.environ.get("PORT", "9400"))
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
SCENARIO = os.environ.get("SCENARIO", "none").lower()
NODE = os.environ.get("NODE_NAME", os.environ.get("HOSTNAME", "gpu-node-0"))
MODEL = os.environ.get("FAKE_GPU_NAME", "NVIDIA A100-SXM4-40GB")
MEM_TOTAL_MIB = int(os.environ.get("FAKE_GPU_MEM_MIB", str(40 * 1024)))

START = time.time()


def metrics_text():
    t = time.time() - START
    lines = [
        "# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (%).",
        "# TYPE DCGM_FI_DEV_GPU_UTIL gauge",
        "# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (MiB).",
        "# TYPE DCGM_FI_DEV_FB_USED gauge",
        "# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free (MiB).",
        "# TYPE DCGM_FI_DEV_FB_FREE gauge",
        "# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (C).",
        "# TYPE DCGM_FI_DEV_GPU_TEMP gauge",
        "# HELP DCGM_FI_DEV_POWER_USAGE Power draw (W).",
        "# TYPE DCGM_FI_DEV_POWER_USAGE gauge",
        "# HELP DCGM_FI_DEV_ECC_DBE_VOL_TOTAL Volatile double-bit ECC errors.",
        "# TYPE DCGM_FI_DEV_ECC_DBE_VOL_TOTAL counter",
        "# HELP DCGM_FI_PROF_PIPE_TENSOR_ACTIVE Tensor-core active ratio.",
        "# TYPE DCGM_FI_PROF_PIPE_TENSOR_ACTIVE gauge",
    ]
    for gpu in range(GPU_COUNT):
        lbl = (f'gpu="{gpu}",UUID="GPU-fake-{gpu:08d}",device="nvidia{gpu}",'
               f'modelName="{MODEL}",Hostname="{NODE}"')

        if SCENARIO == "slow":
            util = 12 + 6 * math.sin(t / 5)          # starved GPU
            tensor = 0.05
            mem_used = MEM_TOTAL_MIB * 0.30
        elif SCENARIO == "cuda_oom":
            # climbs toward full then the job dies and it drops to idle
            frac = min(0.99, 0.5 + t / 60.0)
            util = 95 if frac < 0.99 else 0
            tensor = 0.6 if frac < 0.99 else 0.0
            mem_used = MEM_TOTAL_MIB * frac
        elif SCENARIO == "nccl_timeout":
            util = 100 if t < 8 else 0               # busy-spin then abort
            tensor = 0.0                              # blocked in comms, no compute
            mem_used = MEM_TOTAL_MIB * 0.65
        else:
            util = 88 + 8 * math.sin(t / 3)
            tensor = 0.55 + 0.1 * math.sin(t / 4)
            mem_used = MEM_TOTAL_MIB * 0.70

        ecc = 1 if (SCENARIO == "ecc" and t > 10) else 0
        temp = 38 + util * 0.4
        power = 90 + util * 3.0

        lines += [
            f"DCGM_FI_DEV_GPU_UTIL{{{lbl}}} {util:.0f}",
            f"DCGM_FI_DEV_FB_USED{{{lbl}}} {mem_used:.0f}",
            f"DCGM_FI_DEV_FB_FREE{{{lbl}}} {MEM_TOTAL_MIB - mem_used:.0f}",
            f"DCGM_FI_DEV_GPU_TEMP{{{lbl}}} {temp:.0f}",
            f"DCGM_FI_DEV_POWER_USAGE{{{lbl}}} {power:.1f}",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{lbl}}} {ecc}",
            f"DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{{{lbl}}} {max(0.0, tensor):.3f}",
        ]
    return "\n".join(lines) + "\n"

# The exporter runs a lightweight HTTP server on port 9400. When Prometheus scrapes /metrics, the handler calls
# metrics_text() to generate DCGM-style metrics, returns them in Prometheus exposition format, and Prometheus stores
# them as time-series data for Grafana dashboards and alerting.
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = metrics_text().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence per-request noise
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    print(f"dcgm-fake-exporter listening on :{PORT} "
          f"gpus={GPU_COUNT} scenario={SCENARIO} node={NODE}", flush=True)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
