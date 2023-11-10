#wirehead_manager/bin/activate!/bin/bash

# Launch the two shell scripts
./wirehead_server.sh &
./wirehead_generate.sh &
./wirehead_train.sh

