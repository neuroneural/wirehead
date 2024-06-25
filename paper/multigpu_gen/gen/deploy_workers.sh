#!/bin/bash
#SBATCH --job-name=wh_bnchmk
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./gen/log/out_gen%A_%a.log
#SBATCH --error=./gen/log/err_gen%A_%a.log
#SBATCH --time=06:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH --array=0-10

echo "This is a Synthseg generation job running on node $(hostname)"
echo "Project Name: $PROJECT_NAME"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Error output test" >&2

source /data/users1/mdoan4/wirehead/paper/train/wirehead/bin/activate

# Pass PROJECT_NAME and EXPERIMENT_ID to the Python script
stdbuf -o0 python /data/users1/mdoan4/wirehead/paper/multigpu_gen/gen/worker.py --experiment_id "$EXPERIMENT_ID"
