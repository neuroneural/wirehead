#!/bin/bash

#SBATCH --job-name=wireheadsergey
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH --gres=gpu:A40:1
#SBATCH --output=./log/generate_output_%A_%a.log
#SBATCH --error=./log/generate_error_%A_%a.log
#SBATCH --time=06:00:00
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH --array=0-2

echo "This is a test job running on node $(hostname)"
echo "Error output test" >&2

conda init 
conda activate wirehead_generate

# Function to terminate child processes
terminate_child_processes() {
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$
}

# Function to check if MongoDB is running
check_mongo() {
    if pgrep -x "mongod" > /dev/null
    then
        echo "MongoDB is running."
    else
        echo "MongoDB is not running. Please start mongod according to your system specs"
        kill -SIGINT $$
    fi
}

check_mongo

# Trap the SIGINT signal (Ctrl+C)
trap terminate_child_processes SIGINT

stdbuf -o0 python manager.py &
stdbuf -o0 python generator.py


