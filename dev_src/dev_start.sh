#!/bin/bash

CAP=30

echo $HOSTNAME >&2

# Start redis
/data/users1/mdoan4/wirehead/redis/redis-stable/src/redis-server /data/users1/mdoan4/wirehead/redis/redis-stable/redis.conf >> ./log/manager_wirehead_output.log 2>> ./log/manager_wirehead_errors.log &
REDIS_PID=$!

sleep 2 

LOCAL_IP=$(hostname -I | awk '{print $1}')
echo $LOCAL_IP

# Start python scripts
trap 'pkill -P $$' EXIT
source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead
python /data/users1/mdoan4/wirehead/dev_src/generate.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID &
python /data/users1/mdoan4/wirehead/dev_src/manager.py --ip $LOCAL_IP --cap $CAP $SLURM_ARRAY_TASK_ID


# Cleanup 
kill $REDIS_PID


wait



