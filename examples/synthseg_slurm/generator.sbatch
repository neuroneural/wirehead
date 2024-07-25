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
#SBATCH --array=0-2


#conda init bash
#conda activate wirehead
source /trdapps/linux-x86_64/envs/plis_conda/bin/activate /trdapps/linux-x86_64/envs/plis_conda/envs/wirehead

python generator.py
