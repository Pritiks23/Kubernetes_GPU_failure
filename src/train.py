#!/usr/bin/env python3
"""
Distributed PyTorch training entrypoint for the GPU Platform Failure Investigation Lab.

Design notes (read this in the interview):
  * Real distributed training scaffolding: torch.distributed init, a tiny model,
    a training loop, periodic loss/throughput logging, and graceful rank-0 logging.
  * Runs on CPU with the `gloo` backend so it needs NO physical GPU, but it *reports*
    itself as a GPU job and emits CUDA/NCCL-shaped telemetry so the failure scenarios
    are indistinguishable from a real cluster at the log/observability layer.
  * A FAILURE_MODE env var deterministically injects realistic failure signatures.
    This is what the lab investigates.

Environment variables:
  FAILURE_MODE   one of: none|cuda_oom|nccl_timeout|nccl_mismatch|crash|hang|slow
  RANK           global rank (set by the launcher / k8s)
  WORLD_SIZE     total ranks
  MASTER_ADDR    rendezvous host
  MASTER_PORT    rendezvous port
  EPOCHS         number of epochs (default 50)
  BATCH_SIZE     per-step batch (default 64)
"""
import datetime
import os
import sys
import time

# --- helpers -----------------------------------------------------------------

def env(name, default=None):
    return os.environ.get(name, default)


RANK = int(env("RANK", "0"))
WORLD_SIZE = int(env("WORLD_SIZE", "1"))
LOCAL_RANK = int(env("LOCAL_RANK", "0"))
MASTER_ADDR = env("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = env("MASTER_PORT", "29500")
FAILURE_MODE = env("FAILURE_MODE", "none").lower()
EPOCHS = int(env("EPOCHS", "50"))
BATCH_SIZE = int(env("BATCH_SIZE", "64"))
HOSTNAME = env("HOSTNAME", "unknown")

# Simulated GPU identity so logs look real even on a CPU node.
GPU_NAME = env("FAKE_GPU_NAME", "NVIDIA A100-SXM4-40GB")
GPU_MEM_GB = float(env("FAKE_GPU_MEM_GB", "40"))


def log(msg, level="INFO"):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [rank{RANK}] [{level}] {msg}", flush=True)


def banner():
    log(f"starting trainer on host={HOSTNAME}")
    log(f"world_size={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK}")
    log(f"MASTER_ADDR={MASTER_ADDR} MASTER_PORT={MASTER_PORT}")
    log(f"FAILURE_MODE={FAILURE_MODE} epochs={EPOCHS} batch_size={BATCH_SIZE}")
    # Mimic the CUDA/driver discovery banner real jobs emit.
    log(f"CUDA runtime initialized: device 0 = '{GPU_NAME}' "
        f"({GPU_MEM_GB:.0f} GiB), driver 535.104.05, cuda 12.2")


# --- failure injection -------------------------------------------------------

def maybe_fail_early():
    """Failures that happen before / during distributed init."""
    if FAILURE_MODE == "nccl_timeout":
        # Simulate a rank that never reaches the rendezvous: the others time out.
        log("initializing process group: backend=nccl init_method=env://")
        log(f"NCCL INFO Bootstrap : Using eth0:10.244.{RANK}.7<0>")
        log(f"NCCL INFO NET/Socket : Using [0]eth0:10.244.{RANK}.7<0>")
        log("NCCL INFO Setting affinity for GPU 0 to 0fff")
        if RANK == 0:
            # rank 0 waits for peers that never show up
            log("waiting for all ranks to reach c10d rendezvous "
                "(timeout=600s)...")
            time.sleep(8)  # shortened for the lab; real default is 600s/1800s
            log("Watchdog caught collective operation timeout: "
                "WorkNCCL(OpType=ALLREDUCE, Timeout(ms)=600000) ran for "
                "600351 milliseconds before timing out.", level="ERROR")
            log("[Rank 0] Timeout waiting for ranks [2,3] to join. "
                "Aborting process group.", level="ERROR")
            log("ProcessGroupNCCL.cpp:874 NCCL watchdog thread terminated "
                "with exception", level="ERROR")
            raise SystemExit(1)
        else:
            # peer ranks "hang" then get killed
            time.sleep(8)
            log("torch.distributed.DistBackendError: NCCL communicator "
                "was aborted on rank %d." % RANK, level="ERROR")
            raise SystemExit(1)

    if FAILURE_MODE == "nccl_mismatch":
        log("initializing process group: backend=nccl init_method=env://")
        log(f"NCCL INFO Bootstrap : Using eth0:10.244.{RANK}.7<0>")
        log("NCCL INFO Channel 00/02 :    0   1   2   3")
        log("NCCL WARN NET/IB : No device found.", level="WARN")
        # Classic NCCL topology / mismatch error.
        log("NCCL WARN Mismatched collective detected: rank %d called "
            "ALLREDUCE with count=262144 but rank 0 called ALLREDUCE with "
            "count=131072" % RANK, level="ERROR")
        log("ncclInvalidUsage: This usually reflects an error in the "
            "application (mismatched collective sizes across ranks).",
            level="ERROR")
        log("NCCL error in: ../torch/csrc/distributed/c10d/"
            "ProcessGroupNCCL.cpp:1287, invalid usage (run with "
            "NCCL_DEBUG=WARN for details), NCCL version 2.18.3", level="ERROR")
        raise SystemExit(1)

    if FAILURE_MODE == "image_pull":
        # This mode is never actually reached: the scenario manifest uses a
        # bad image tag so the kubelet fails before the container starts.
        # Kept here only as documentation.
        pass


def maybe_fail_during_training(step, global_step):
    if FAILURE_MODE == "cuda_oom" and global_step == 12:
        # Realistic CUDA OOM traceback.
        alloc = 39.4
        reserved = 39.5
        log("Trying to allocate 2.00 GiB. GPU 0 has a total capacity of "
            f"{GPU_MEM_GB:.2f} GiB of which 312.00 MiB is free.", level="ERROR")
        sys.stderr.write(
            "Traceback (most recent call last):\n"
            '  File "/workspace/train.py", line 211, in <module>\n'
            "    loss.backward()\n"
            '  File "/usr/local/lib/python3.10/site-packages/torch/_tensor.py",'
            " line 522, in backward\n"
            "    torch.autograd.backward(self, gradient, ...)\n"
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
            f"allocate 2.00 GiB. GPU 0 has a total capacity of {GPU_MEM_GB:.2f} "
            f"GiB of which 312.00 MiB is free. Process 1 has {alloc:.2f} GiB "
            "memory in use. Of the allocated memory %0.2f GiB is allocated by "
            "PyTorch, and 268.00 MiB is reserved by PyTorch but unallocated. "
            "If reserved but unallocated memory is large try setting "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid "
            "fragmentation.\n" % reserved)
        sys.stderr.flush()
        # Exit 1; k8s will mark the container failed (then CrashLoopBackOff).
        raise SystemExit(1)

    if FAILURE_MODE == "crash" and global_step == 5:
        # Segfault-style abrupt death -> exit code 139 surfaces in k8s.
        log("CUDA error: an illegal memory access was encountered", level="ERROR")
        log("CUDA kernel errors might be asynchronously reported at some other "
            "API call, so the stacktrace below might be incorrect.", level="ERROR")
        os._exit(139)

    if FAILURE_MODE == "hang" and global_step == 7:
        # Deadlock: one rank stops issuing collectives. Others block in allreduce.
        if RANK == 0:
            log("entering long synchronous checkpoint (simulated stall)...")
            while True:
                time.sleep(30)
                log("still inside checkpoint... (rank 0 not calling allreduce)")
        else:
            log("blocked in c10d allreduce waiting on rank 0...", level="WARN")
            while True:
                time.sleep(30)
                log("NCCL INFO still waiting for rank 0 in ALLREDUCE "
                    "(no progress)", level="WARN")


# --- main --------------------------------------------------------------------

def main():
    banner()
    maybe_fail_early()

    # Real torch.distributed init when torch is available; degrade gracefully
    # so the failure-log scenarios still run in minimal images.
    dist = None
    try:
        import torch
        import torch.distributed as dist_mod
        import torch.nn as nn
        dist = dist_mod
        backend = "gloo"  # CPU-friendly; logs still say "nccl-style"
        log(f"initializing process group: backend={backend} init_method=env://")
        if WORLD_SIZE > 1:
            dist.init_process_group(
                backend=backend,
                timeout=datetime.timedelta(seconds=60),
                world_size=WORLD_SIZE,
                rank=RANK,
            )
            log(f"process group ready: rank {RANK}/{WORLD_SIZE}")
            dist.barrier()
            log("post-init barrier passed; all ranks present")

        model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        have_torch = True
    except ImportError:
        log("torch not present in image; running in log-only mode", level="WARN")
        have_torch = False

    global_step = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        for step in range(20):
            maybe_fail_during_training(step, global_step)
            if have_torch:
                x = torch.randn(BATCH_SIZE, 256)
                y = torch.randint(0, 10, (BATCH_SIZE,))
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                if dist is not None and WORLD_SIZE > 1:
                    for p in model.parameters():
                        dist.all_reduce(p.grad)
                        p.grad /= WORLD_SIZE
                opt.step()
                loss_val = float(loss.item())
            else:
                loss_val = max(0.05, 2.5 * (0.97 ** global_step))
                time.sleep(0.2)

            if FAILURE_MODE == "slow":
                time.sleep(1.5)  # straggler / data-loader bottleneck

            if global_step % 5 == 0:
                imgs_s = BATCH_SIZE * WORLD_SIZE / max(1e-6, (time.time() - t0) / (global_step + 1))
                util = 18 if FAILURE_MODE == "slow" else 94
                log(f"epoch={epoch} step={step} global_step={global_step} "
                    f"loss={loss_val:.4f} throughput={imgs_s:.0f} img/s "
                    f"gpu_util={util}% gpu_mem={int(GPU_MEM_GB*0.7*1024)}MiB")
            global_step += 1

    log("training complete; exiting 0")
    if dist is not None and WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
