#!/bin/bash

PROJECT_NAME="wirehead_1xA100_wirehead"
EXPERIMENT_ID=$(date +"%Y-%m-%d%H-%M")


# Run the training script with these hardware configs
srun -p qTRDGPUH -A psy53c17 -v -t600 -N1-1 -c16 --gres=gpu:A100:1 --mem=200g --pty test.sh --project_name wirehead_dev --experiment_id hi
