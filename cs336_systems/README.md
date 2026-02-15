1. 网站 : https://hpcgame.pku.edu.cn/org/2df8f692-0316-4682-80cd-591732b1eda6/contest/d61cbf2d-b554-43bf-a3de-aaa0d46264d9

2. `slurm`提交任务
```bash
nvidia-smi # 查看所有卡的状态, 一般会看到8张卡
srun -t 00:10:00 --gres=gpu:1 --cpus-per-task=1 --mem=8G --pty /bin/bash
# 申请 1 张 GPU, 限时10 min
nvidia-smi # 确认身份, 只会看到属于你的那张卡
exit # 结束后退出
```

3. 实验
```bash
uv run /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=profile_xl_128_fwd \
  --force-overwrite=true \
  python -m cs336_systems.benchmark --model_size xl --context_length 128 --mode forward --n_steps 20
```