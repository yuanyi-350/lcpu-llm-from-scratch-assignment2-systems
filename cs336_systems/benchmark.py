import torch
import timeit
import argparse
import numpy as np
from einops import rearrange, einsum
from cs336_basics.model import BasicsTransformerLM
import cs336_basics.model
import torch.cuda.nvtx as nvtx
import math
import contextlib

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "n_layers": 12, "n_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "n_layers": 24, "n_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "n_layers": 36, "n_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "n_layers": 48, "n_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "n_layers": 32, "n_heads": 32},
}

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    
    with nvtx.range("computing attention scores"):
        # (Batch, Heads, Seq, d_k) @ (Batch, Heads, d_k, Seq) -> (Batch, Heads, Seq, Seq)
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))


    with nvtx.range("computing softmax"):
        attention_weights = torch.softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        attention_output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return attention_output

def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # [Monkey Patching] 
    if hasattr(cs336_basics.model, 'scaled_dot_product_attention'):
        print("Monkey patching scaled_dot_product_attention with NVTX annotated version...")
        
        original_attn = cs336_basics.model.scaled_dot_product_attention
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    else:
        print("Warning: Could not find scaled_dot_product_attention in cs336_basics.model to patch.")

    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["n_layers"],
        num_heads=config["n_heads"],
        rope_theta=10000.0,
    ).to(device)

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters())

    batch_size = 4
    x = torch.randint(0, 10000, (batch_size, args.context_length), device=device)

    if args.mixed_precision:
        print("Using Mixed Precision (BFloat16)")
        mp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        print("Using Full Precision (FP32)")
        mp_context = contextlib.nullcontext()

    if args.mode == "forward":
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"Warming up for {args.warmup_steps} steps...")
    with nvtx.range("Warmup Phase"):
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            
            with mp_context:
                out = model(x)
                if args.mode in ["backward", "optimizer"]:
                    loss = out.sum()

            if args.mode in ["backward", "optimizer"]:
                loss.backward()
                if args.mode == "optimizer":
                    optimizer.step()
            sync()

    print(f"Measuring for {args.n_steps} steps...")
    times = []

    if args.profile_memory:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("Starting memory recording...")
        torch.cuda.memory._record_memory_history(max_entries=100000)

    for i in range(args.n_steps):
        if args.mode in ["backward", "optimizer"]:
             optimizer.zero_grad(set_to_none=True)

        sync()
        start_time = timeit.default_timer()

        with nvtx.range(f"Step {i}"):
            with nvtx.range("Forward Pass"):
                with mp_context:
                    out = model(x)
                    if args.mode in ["backward", "optimizer"]:
                        loss = out.sum()

            if args.mode in ["backward", "optimizer"]:
                with nvtx.range("Backward Pass"):
                    loss.backward()

            if args.mode == "optimizer":
                with nvtx.range("Optimizer Step"):
                    optimizer.step()

        sync()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    
    if args.profile_memory:
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak Memory Usage: {peak_mem:.2f} MB")

        snapshot_file = f"memory_snapshot_{args.model_size}_{args.context_length}_{args.mode}_{
            'mp' if args.mixed_precision else 'fp32'}.pickle"
        try:
            torch.cuda.memory._dump_snapshot(snapshot_file)
            print(f"Memory snapshot saved to {snapshot_file}")
        except Exception as e:
            print(f"Failed to dump snapshot: {e}")
        
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Model: {args.model_size}, Context: {args.context_length}, Mode: {args.mode}, BF16: {args.mixed_precision}")
    print(f"Average Time: {np.mean(times):.4f} s")
    print(f"Std Dev: {np.std(times):.4f} s")

    if hasattr(cs336_basics.model, 'scaled_dot_product_attention'):
        cs336_basics.model.scaled_dot_product_attention = original_attn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, choices=MODEL_CONFIGS.keys(), default="small", help="Model size from Table 1")
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--mode", type=str, choices=["forward", "backward", "optimizer"], default="forward", help="Measure forward, forward+backward, or full training step")
    
    parser.add_argument("--mixed_precision", action="store_true", help="Enable Mixed Precision (BF16)")
    parser.add_argument("--profile_memory", action="store_true", help="Enable memory profiling and snapshot dump")
    parser.add_argument("--compile", action="store_true", help="")

    args = parser.parse_args()
    benchmark(args)