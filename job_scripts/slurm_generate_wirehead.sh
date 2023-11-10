#!/bin/bash

#SBATCH --job-name=wirehead-train-test
#SBATCH --nodes=4
#SBATCH -c 16 
#SBATCH --mem=10g
#SBATCH --output=./log/generate_output.log
#SBATCH --error=./log/generate_error.log
#SBATCH --time=01:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17


PORT=6379
LOCAL_IP='arctrdagn041'

# Start python scripts
trap 'pkill -P $$' EXIT
source /data/users1/mdoan4/wirehead/envs/wirehead_generate/bin/activate

python /data/users1/mdoan4/wirehead/src/generate.py --ip $LOCAL_IP $SLURM_ARRAY_TASK_ID  

echo "Generator: Terminated"

wait

