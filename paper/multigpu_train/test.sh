#!/bin/bash

# Declare project name
PROJECT_NAME="wirehead_1xA100_wirehead"

# Generate experiment ID
EXPERIMENT_ID=$(date +"%Y-%m-%d_%H-%M")

echo Running experiment $PROJECT_NAME/$EXPERIMENT_ID

# Function to terminate child processes
terminate_child_processes() {
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$
}

# Trap the SIGINT signal (Ctrl+C)
trap terminate_child_processes SIGINT

# Run Python scripts with project name and experiment ID as arguments
python clean.py
python manager.py &
python generator.py "$PROJECT_NAME" "$EXPERIMENT_ID" &
python multigpu_train.py "$PROJECT_NAME" "$EXPERIMENT_ID"
wait
kill -SIGINT $$
