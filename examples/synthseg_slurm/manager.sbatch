#!/bin/bash

#SBATCH --job-name=wirehead
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem=10g
#SBATCH --gres=gpu:A40:0
#SBATCH --output=./log/manager%A_%a.log
#SBATCH --error=./log/manager%A_%a.err
#SBATCH --time=10:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17

echo "This is a wirehead job on $(hostname)"

conda init bash
conda activate wirehead

python manager.py
