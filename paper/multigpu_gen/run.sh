#!/bin/bash

PROJECT_NAME="wirehead_1xA100_multigpugen"
EXPERIMENT_ID=$(date +"%Y-%m-%d_%H-%M")

export PROJECT_NAME=$PROJECT_NAME
export EXPERIMENT_ID=$EXPERIMENT_ID

cat gen/deploy_workers.sh
cat conf/wirehead_config.yaml

# Run the training script with these hardware configs
srun -p qTRDGPUH -A psy53c17 -v -t600 -N1-1 -c16 --gres=gpu:A100:1 --mem=200g --pty \
 train_for_distributed.sh --project_name $PROJECT_NAME --experiment_id $EXPERIMENT_ID &

# Run the generator and manager with the same configs
sbatch --export=ALL gen/deploy_workers.sh
