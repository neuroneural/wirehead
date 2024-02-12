#!/bin/bash

#SBATCH --job-name=wirehead-train-test
#SBATCH --nodes=1
#SBATCH --output=./log/server_output.log
#SBATCH --error=./log/server_error.log
#SBATCH --time=02:00:00
#SBATCH --nodelist=arctrdagn041
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17

PORT=6379

export PATH=$PATH:/data/users1/mdoan4/wirehead/dependencies/redis/redis-stable/src/

echo $HOSTNAME >&2

# Start redis
/data/users1/mdoan4/wirehead/dependencies/redis/redis-stable/src/redis-server /data/users1/mdoan4/wirehead/src/utils/redis.conf >> ./log/server_wirehead.log 2>> ./log/server_wirehead_error.log &
REDIS_PID=$!

sleep 2 

redis-cli flushall
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo $LOCAL_IP

# Start python scripts
trap 'pkill -P $$' EXIT

source /data/users1/mdoan4/wirehead/envs/wirehead_manager/bin/activate

echo "Wirehead Server: pyenv activated successfully"

python /data/users1/mdoan4/wirehead/src/manager.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID 

echo "Wirehead Server: Terminated"
# Cleanup 
kill $REDIS_PID

wait

