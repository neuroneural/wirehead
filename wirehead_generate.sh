#!/bin/bash

PORT=6379
LOCAL_IP='arctrdagn019'

# Start python scripts
trap 'pkill -P $$' EXIT
source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh

conda activate wirehead_generate
python /data/users1/mdoan4/wirehead/src/generate.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID  

echo "Generator: Terminated"

wait

