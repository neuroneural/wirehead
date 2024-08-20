#!/bin/bash
#SBATCH --job-name=wh-gen
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=200g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./log/generate_output_%A_%a.log
#SBATCH --error=./log/generate_error_%A_%a.log
#SBATCH --time=00:10:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH --array=0-3

NUM_GENERATORS=4
conda activate wirehead # replace with proper activation functions for your environment

for i in $(seq 0 $((NUM_GENERATORS-1))); do
  python generator.py 4 $NUM_GENERATORS $i &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait $pid
done
