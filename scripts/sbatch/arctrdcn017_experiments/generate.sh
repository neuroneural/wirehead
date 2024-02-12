#!/bin/bash

#SBATCH --job-name=wirehead-train-test
#SBATCH --nodes=1
#SBATCH -c 16 
#SBATCH --mem=50g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./log/generate_output.log
#SBATCH --error=./log/generate_error.log
#SBATCH --time=06:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17


PORT=6379
LOCAL_IP='arctrdcn017'
TRAINING_SEG=$1 # example: "training_seg_01.nii.gz"
GENERATOR_LENGTH=800

# Start python scripts
trap 'pkill -P $$' EXIT

source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead_generate


python /data/users1/mdoan4/wirehead/src/generate.py --ip $LOCAL_IP --training_seg $TRAINING_SEG  $SLURM_ARRAY_TASK_ID  

echo "Generator: Terminated"

wait

