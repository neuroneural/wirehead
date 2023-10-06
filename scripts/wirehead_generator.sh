#!/bin/bash
#### These two lines tells wirehead how many GPUs to ask for ####
#SBATCH -N 4
#SBATCH -n 4
#### ------------------------------------------------------- ####
#### Customize your generation node information here         ####
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH --gres=gpu:A40:1
#SBATCH -p qTRDGPU
#SBATCH -t 4-00
#### ------------------------------------------------------- ####
#SBATCH -J wirehead-generate
#SBATCH -e ./log/wirehead-%A.err
#SBATCH -A psy53c17

sleep 10s
echo $HOSTNAME >&2
source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead
python /data/users1/mdoan4/wirehead/src/generate.py $SLURM_ARRAY_TASK_ID
wait
