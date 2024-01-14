#wirehead_manager/bin/activate!/bin/bash

# Launch the two shell scripts
trap 'pkill -P $$' EXIT
./server.sh &
./generate.sh &
./train.sh



