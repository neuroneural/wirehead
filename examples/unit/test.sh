#!/bin/bash

run_clean() {
    python clean.py
}
# Function to run test.py
run_test() {
    python test.py
}

# Function to run loader.py
run_loader() {
    python loader.py
    # Send a signal to the test.py process to terminate it
    pkill -f test.py
}

# Run test.py in the background
run_clean
run_test &
test_pid=$!

# Run loader.py in the foreground
run_loader

# Wait for loader.py to finish
wait $!

# Print a message indicating that both scripts have finished
echo
echo
echo "-----------"
echo "Test passed"
echo "-----------"
