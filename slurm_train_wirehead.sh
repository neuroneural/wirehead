#!/bin/bash

#SBATCH --job-name=wirehead-train-test
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH --output=./log/wirehead_training_test_output.log
#SBATCH --error=./log/wirehead_training_test_errors.log
#SBATCH --time=01:00:00
#SBATCH --nodelist=arctrdagn041
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17

PORT=6379

export PATH=$PATH:/data/users1/mdoan4/wirehead/redis/redis-stable/src/

echo $HOSTNAME >&2

# Start redis
/data/users1/mdoan4/wirehead/redis/redis-stable/src/redis-server /data/users1/mdoan4/wirehead/dev_src/redis.conf >> ./log/manager_wirehead_output.log 2>> ./log/manager_wirehead_errors.log &
REDIS_PID=$!

sleep 2 

redis-cli flushall
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo $LOCAL_IP

# Start python scripts
trap 'pkill -P $$' EXIT

source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh

conda activate wirehead_train
python /data/users1/mdoan4/wirehead/dev_src/manager.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID &
python /data/users1/mdoan4/wirehead/dev_src/dataloader.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID  

echo "murdered"
# Cleanup 
kill $REDIS_PID

wait

