#!/bin/bash
#SBATCH -J allreduce_bench
#SBATCH --gres=gpu:4              # 改成 2/4/6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export XDG_CACHE_HOME=$HOME/.cache

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + (${SLURM_JOB_ID} % 20000)))

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $HOME/cs336-2

WORLD_SIZE=4
BYTES=$((100*1024*1024))   # 1MB/10MB/100MB/1GB 自己改
ITERS=30
WARMUP=5

srun --ntasks=1 --gres=gpu:${WORLD_SIZE} \
  uv run python -m cs336_systems.distributed_communication_single_node \
    --backend nccl \
    --world_size ${WORLD_SIZE} \
    --bytes ${BYTES} \
    --iters ${ITERS} \
    --warmup ${WARMUP} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
