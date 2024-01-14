#!/bin/bash

# Wrapper script for wirehead train and test sbatch scripts

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <NODE_NAME_OR_IP_TO_HOST_REDIS_AND_TRAIN_ON>"
    exit 1
fi

NODE_NAME="$1"

# Call sbatch scripts with substituted node name
sbatch <(sed "s/arctrdagn041/$NODE_NAME/g" ./generate_sbatch.sh)
sbatch <(sed "s/arctrdagn041/$NODE_NAME/g" ./server_sbatch.sh)
sbatch <(sed "s/arctrdagn041/$NODE_NAME/g" ./train_sbatch.sh)
