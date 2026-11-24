#!/bin/sh
# One-GPU check for ROQUO: print where the job ran and the GPU it was given, then exit with
# the status given as the first argument (default 0). Exits 2 if no GPU can be listed.
set -u
echo "hello from the roquo prefect gpu demo"
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; then
  echo "No GPU could be listed: check the GPU request and that nvidia-smi is on PATH." >&2
  exit 2
fi
exit "${1:-0}"
