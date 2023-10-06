#!/bin/bash

# Perhaps declare some defaults here

## Node to use dataloader and redis on
## How many GPUS to use
## What should the cap be for the rotating databases

# List of Slurm scripts to execute
declare -a scripts=("wirehead_generator.sh" "wirehead_manager.sh")

# Loop to execute each script
for script in "${scripts[@]}"; do
    sbatch $script
done

