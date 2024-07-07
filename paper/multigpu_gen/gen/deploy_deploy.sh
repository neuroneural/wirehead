#!/bin/bash

export PROJECT_NAME="wirehead_dev"
export EXPERIMENT_ID="test"

sbatch --export=ALL ./gen/deploy_workers.sh

