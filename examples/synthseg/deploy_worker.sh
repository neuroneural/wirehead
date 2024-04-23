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

echo "This is a test job running on node $(hostname)"
echo "Error output test" >&2

export PYTHONPATH=/data/users1/mdoan4/wirehead:$PYTHONPATH
source /trdapps/linux-x86_64/envs/plis_conda/bin/activate /trdapps/linux-x86_64/envs/plis_conda/envs/synthseg_38
stdbuf -o0 python worker.py
