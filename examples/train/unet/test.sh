#!/bin/bash

terminate_child_processes() {
    pkill -SIGTERM -P $$
}

# Trap the SIGINT signal (Ctrl+C)
trap terminate_child_processes SIGINT

python clean.py

python utils/manager.py &

python generator.py &

python train.py

kill -SIGINT $$
