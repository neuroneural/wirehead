#!/bin/bash

# Function to terminate child processes and SLURM job
terminate_processes() {
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$

    # Cancel the SLURM job if job_id is set
    if [ ! -z "$job_id" ]; then
        scancel $job_id
        echo "Cancelled SLURM job $job_id"
    fi
}

# Trap the SIGINT signal (Ctrl+C)
trap terminate_processes SIGINT

python clean.py

# Run sbatch and capture the job ID
job_id=$(sbatch --parsable deploy_worker.sh)
echo "Started SLURM job with ID: $job_id"

sleep 30
python loader.py

# Terminate all processes (including the SLURM job)
terminate_processes

# Exit the script
exit 0
