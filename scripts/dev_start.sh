#!/bin/bash

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
# Activate the wirehead environment and run the scripts
source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh

conda activate wirehead_generate
which python
/data/users1/mdoan4/anaconda3/envs/wirehead/bin/python /data/users1/mdoan4/wirehead/dev_src/redis_status.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID 
/data/users1/mdoan4/anaconda3/envs/wirehead/bin/python /data/users1/mdoan4/wirehead/dev_src/generate.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID &
/data/users1/mdoan4/anaconda3/envs/torch2/bin/python /data/users1/mdoan4/wirehead/dev_src/manager.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID 

# Activate the torch2 environment and run the scripts
#conda activate torch2
#/data/users1/mdoan4/anaconda3/envs/torch2/bin/python /data/users1/mdoan4/wirehead/dev_src/dataloader.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID 

sleep 10
# Cleanup 
kill $REDIS_PID

wait
  
