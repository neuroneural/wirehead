#!/bin/bash

# Function to terminate child processes
terminate_child_processes() {
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$
}
# Trap the SIGINT signal (Ctrl+C)
trap terminate_child_processes SIGINT

python clean.py

sbatch deploy_worker.sh &

sleep 30

python loader.py

kill -SIGINT $$
