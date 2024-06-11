#!/bin/bash

#SBATCH --job-name=wireheadsergey
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./log/generate_output_%A_%a.log
#SBATCH --error=./log/generate_error_%A_%a.log
#SBATCH --time=06:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH --array=0-2

echo "This is a Synthseg generation job running on node $(hostname)"
echo "Error output test" >&2

conda init 
conda activate wirehead_generate

stdbuf -o0 python worker.py
