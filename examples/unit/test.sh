#!/bin/bash

trap terminate_child_processes SIGINT

python clean.py

python manager.py &

python generator.py &

python loader.py

kill -SIGINT $$
