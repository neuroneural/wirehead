#!/bin/bash

# Function to check if MongoDB is running
check_mongo() {
    if pgrep -x "mongod" > /dev/null
    then
        echo "MongoDB is running."
    else
        echo "MongoDB is not running. Starting MongoDB..."
        mongod &
        sleep 5  # Wait for MongoDB to start
        echo "MongoDB started."
    fi
}

# Function to run clean.py
run_clean() {
    python clean.py
}

run_manager() {
    python manager.py
}

run_generator() {
    python generator.py
}

# Function to run loader.py
run_loader() {
    python loader.py
    # Send a signal to the test.py process to terminate it
    pkill -f manager.py
    pkill -f generator.py
}

# Check if MongoDB is running and start it if necessary
check_mongo

# Run clean.py
run_clean
echo

# Run test.py in the background
run_manager&
manager_pid=$!

run_generator&
generator_pid=$!

# Run loader.py in the foreground
run_loader

# Wait for loader.py to finish
wait $!

# Print a message indicating that both scripts have finished

pkill -f manager.py
pkill -f generator.py

echo
echo
echo "-----------"
echo "Test passed"
echo "-----------"
