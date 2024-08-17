#!/bin/bash

# Function to terminate child processes
terminate_child_processes() {
    # Send SIGTERM to all child processes
    pkill -SIGTERM -P $$
}

trap terminate_child_processes SIGINT

python clean.py

python generator.py &

python loader.py

kill -SIGINT $$
