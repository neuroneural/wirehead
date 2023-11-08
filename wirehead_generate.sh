#!/bin/bash

PORT=6379
LOCAL_IP='localhost'

# Start python scripts
trap 'pkill -P $$' EXIT
source /data/users1/mdoan4/wirehead/envs/wirehead_generate/bin/activate

python /data/users1/mdoan4/wirehead/src/generate.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID  

echo "Generator: Terminated"

wait

