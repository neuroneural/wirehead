#!/bin/bash
#
#
LOCAL_IP="localhost"
echo "redis started successfully"

# Start python scripts
trap 'pkill -P $$' EXIT

source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead_train
echo "conda activated successfully"
python /data/users1/mdoan4/wirehead/tests/test_training.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID  

echo "test was successful"
#
