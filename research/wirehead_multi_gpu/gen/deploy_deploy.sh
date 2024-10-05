#!/bin/bash

export PROJECT_NAME="wirehead_dev"
export EXPERIMENT_ID="2xgen"

sbatch --export=ALL ./gen/deploy_workers.sh

