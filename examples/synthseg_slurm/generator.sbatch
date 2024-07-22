#!/bin/bash

#SBATCH --job-name=wirehead
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./log/generate_output_%A_%a.log
#SBATCH --error=./log/generate_error_%A_%a.log
#SBATCH --time=00:15:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH --array=0-999%5


conda init bash
conda activate wirehead

python generatorX8.py
