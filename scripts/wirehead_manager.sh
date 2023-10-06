#!/bin/bash
#SBATCH --job-name=wirehead_manager_job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=5G
#SBATCH -t 8-00
#SBATCH -J manager-wirehead
#SBATCH -p qTRDBF 
#SBATCH --nodelist=arctrdcn019
#SBATCH -e ./log/manager_wirehead-%A.err
#SBATCH -A psy53c17

# Start Redis with a specific configuration file
/data/users1/mdoan4/wirehead/redis/redis-stable/src/redis-server /data/users1/mdoan4/wirehead/redis/redis-stable/redis.conf &  # Replace with your actual path
REDIS_PID=$!

# Wait a bit for Redis to initialize (optional)
sleep 10

source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead
python /data/users1/mdoan4/wirehead/src/manager.py $SLURM_ARRAY_TASK_ID

# Something something dataloader right here
# python /path/to/dataloader(or training)/script
sleep 10000

# Kill the Redis server after the Python script finishes
kill $REDIS_PID

wait
