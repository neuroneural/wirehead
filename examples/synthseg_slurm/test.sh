#!/bin/bash

# Function to terminate child processes and SLURM job
terminate_processes() {
    # remove garbage from tensorflow 
    rm __autograph*
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$

    # Cancel the SLURM job if job_id is set
    if [ ! -z "$generator_id" ]; then
        scancel $generator_id
        echo "Cancelled SLURM job $generator_id"
    fi
    if [ ! -z "$manager_id" ]; then
        scancel $manager_id
        echo "Cancelled SLURM job $manager_id"
    fi
}

# Trap the SIGINT signal (Ctrl+C)
trap terminate_processes SIGINT

python clean.py

# Run sbatch and capture the job ID
generator_id=$(sbatch --parsable generator.sbatch)
echo "Started SLURM job with ID: $generator_id"
manager_id=$(sbatch --parsable manager.sbatch)
echo "Started SLURM job with ID: $manager_id"

python loader.py
echo "Swap occurred, ready to train whenever!"
sleep 1

# Terminate all processes (including the SLURM job)
# python clean.py
terminate_processes

# Exit the script
exit 0
