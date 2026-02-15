```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark
Running on cuda
Warming up for 5 steps...
Measuring for 100 steps...
Average Time: 0.0126 s
Std Dev: 0.0002 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size "medium"
Running on cuda
Warming up for 5 steps...
Measuring for 100 steps...
Average Time: 0.0257 s
Std Dev: 0.0075 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size "large"
Running on cuda
Warming up for 5 steps...
Measuring for 100 steps...
Average Time: 0.0381 s
Std Dev: 0.0078 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size "xl"
Running on cuda
Warming up for 5 steps...
Measuring for 100 steps...
Average Time: 0.0598 s
Std Dev: 0.0074 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size "2.7B"
Running on cuda
Warming up for 5 steps...
Measuring for 100 steps...
Average Time: 0.0812 s
Std Dev: 0.0055 s
```

---

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size xl --context_length 128 --mode forward --n_steps 
100
Running on cuda
Monkey patching scaled_dot_product_attention with NVTX annotated version...
Using Full Precision (FP32)
Warming up for 5 steps...
Measuring for 100 steps...
Model: xl, Context: 128, Mode: forward, BF16: False
Average Time: 0.0588 s
Std Dev: 0.0083 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size xl --context_length 128 --mode forward --n_steps 100 --mixed_precision
Running on cuda
Monkey patching scaled_dot_product_attention with NVTX annotated version...
Using Mixed Precision (BFloat16)
Warming up for 5 steps...
Measuring for 100 steps...
Model: xl, Context: 128, Mode: forward, BF16: True
Average Time: 0.0537 s
Std Dev: 0.0076 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size 2.7B --context_length 1024 --mode forward --n_ste
ps 100 --mixed_precision
Running on cuda
Monkey patching scaled_dot_product_attention with NVTX annotated version...
Using Mixed Precision (BFloat16)
Warming up for 5 steps...
Measuring for 100 steps...
Model: 2.7B, Context: 1024, Mode: forward, BF16: True
Average Time: 0.2802 s
Std Dev: 0.0003 s
```

```bash
stu2400010766@lfs-dev:~/cs336-2$ uv run python -m cs336_systems.benchmark --model_size 2.7B --context_length 1024 --mode forward --n_steps 100
Running on cuda
Monkey patching scaled_dot_product_attention with NVTX annotated version...
Using Full Precision (FP32)
Warming up for 5 steps...
Measuring for 100 steps...
Model: 2.7B, Context: 1024, Mode: forward, BF16: False
Average Time: 0.7019 s
Std Dev: 0.0042 s
```