#!/bin/bash

#SBATCH --job-name=wirehead
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=128g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./logs/generate_output_%A_%a.log
#SBATCH --error=./logs/generate_error_%A_%a.log
#SBATCH --time=00:05:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH --array=0-1000%10


NUM_GENERATORS=8
conda init bash
conda activate wirehead

for i in $(seq 0 $((NUM_GENERATORS-1))); do
  python generator.py
done

for pid in "${pids[@]}"; do
  wait $pid
done
