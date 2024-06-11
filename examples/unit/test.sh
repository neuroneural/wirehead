#!/bin/bash

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

python clean.py

python manager.py &

python generator.py &

python loader.py

# Print a message indicating that both scripts have finished
echo
echo "-----------"
echo "Test passed"
echo "-----------"

kill -SIGINT $$
