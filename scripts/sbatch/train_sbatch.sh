#!/bin/bash

#SBATCH --job-name=wirehead-train-test
#SBATCH --nodes=1
#SBATCH --output=./log/training_output.log
#SBATCH --error=./log/training_error.log
#SBATCH --gres=gpu:A40:2
#SBATCH --time=01:00:00
#SBATCH --nodelist=arctrdagn041
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17

PORT=6379
LOCAL_IP='localhost'
trap 'pkill -P $$' EXIT

source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead_train

echo "Wirehead Train: conda activated successfully"

python /data/users1/mdoan4/wirehead/src/train.py $SLURM_ARRAY_TASK_ID 

echo "Wirehead Training: Terminated"
# Cleanup 
kill $REDIS_PID

wait

