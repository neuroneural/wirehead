#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --project_name <project_name> --experiment_id <experiment_id>"
    exit 1
}

# Initialize variables
PROJECT_NAME=""
EXPERIMENT_ID=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --experiment_id)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Check if both arguments are provided
if [ -z "$PROJECT_NAME" ] || [ -z "$EXPERIMENT_ID" ]; then
    usage
fi

echo "Running experiment $PROJECT_NAME/$EXPERIMENT_ID"

# Function to terminate child processes
terminate_child_processes() {
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$
}

# Trap the SIGINT signal (Ctrl+C)
trap terminate_child_processes SIGINT

# Run Python scripts with project name and experiment ID as arguments
python train.py --experiment_name "$EXPERIMENT_ID"

wait
kill -SIGINT $$
