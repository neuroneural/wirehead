#!/bin/bash

#SBATCH --job-name=wirehead-train-test
#SBATCH --nodes=1
#SBATCH -c 16 
#SBATCH --mem=10g
#SBATCH --output=./log/wirehead_generator_test.log
#SBATCH --error=./log/wirehead_generator_errors.log
#SBATCH --time=01:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17

export PATH=$PATH:/data/users1/mdoan4/wirehead/redis/redis-stable/src/

LOCAL_IP='arctrdagn041' # Pass the IP of the training server as an argument

# Start python scripts
trap 'pkill -P $$' EXIT

source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead
python /data/users1/mdoan4/wirehead/dev_src/redis_status.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID &
python /data/users1/mdoan4/wirehead/dev_src/generate.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID 

