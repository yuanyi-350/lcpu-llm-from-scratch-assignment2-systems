import os
import time
import argparse
import json
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank: int, world_size: int, backend: str, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if backend == "nccl":
        # one GPU per rank
        torch.cuda.set_device(rank)

def bytes_to_numel(nbytes: int, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    assert nbytes % element_size == 0, "bytes must be divisible by element size"
    return nbytes // element_size

def bench_worker(rank: int, world_size: int, args):
    setup(rank, world_size, args.backend, args.master_addr, args.master_port)

    device = torch.device("cpu")
    if args.backend == "nccl":
        device = torch.device(f"cuda:{rank}")

    dtype = torch.float32
    numel = bytes_to_numel(args.bytes, dtype)

    # Use a 1-D tensor; shape doesn't matter for bandwidth much
    x = torch.randn((numel,), device=device, dtype=dtype)

    # Optional: touch to avoid lazy alloc effects
    x.mul_(1.0)

    # Warmup
    for _ in range(args.warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times_ms = []
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    # Gather timings to rank0 (object gather is easiest)
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, times_ms)

    if rank == 0:
        # Flatten: list of lists -> all samples
        all_samples = [t for per_rank in gathered for t in per_rank]
        all_samples_sorted = sorted(all_samples)
        mean = sum(all_samples) / len(all_samples)
        median = all_samples_sorted[len(all_samples_sorted) // 2]

        # Estimate effective bandwidth (very rough):
        # Each all-reduce moves ~ 2*(p-1)/p * bytes per rank for ring (order-of-magnitude).
        p = world_size
        algo_factor = 2.0 * (p - 1) / p
        bw_gbps = (algo_factor * args.bytes) / (median / 1000.0) / 1e9  # GB/s

        record = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "backend": args.backend,
            "device": device.type,
            "world_size": world_size,
            "bytes": args.bytes,
            "size_mb": args.bytes / (1024**2),
            "iters": args.iters,
            "warmup": args.warmup,
            "mean_ms": mean,
            "median_ms": median,
            "est_bw_GBps": bw_gbps,
        }

        jsonl_path = "results/distributed_communication_single_node.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[Saved] JSONL appended: {jsonl_path}")
            print(f"[RESULT] {record}")

    dist.destroy_process_group()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    ap.add_argument("--world_size", type=int, choices=[2, 4, 6], required=True)
    ap.add_argument("--bytes", type=int, required=True, help="tensor size in bytes (float32)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--master_addr", type=str, default="127.0.0.1")
    ap.add_argument("--master_port", type=str, default="29437")
    args = ap.parse_args()

    if args.backend == "nccl":
        assert torch.cuda.is_available(), "CUDA required for NCCL"
        ngpu = torch.cuda.device_count()
        assert args.world_size <= ngpu, f"need >= {args.world_size} GPUs, found {ngpu}"

    mp.spawn(
        fn=bench_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True,
    )

if __name__ == "__main__":
    main()
