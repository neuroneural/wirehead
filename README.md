# wirehead #

Caching system for horizontal scaling of synthetic data generators using Redis/MongoDB

---

## Usage ##

- Create a generator that yields a (sample, label) pair
- Either use the default preprocessing options in generate_and_insert() or customize your own
- Deploy to slurm 

```
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
#SBATCH --array=0-20

echo "This is a test job running on node $(hostname)"
echo "Error output test" >&2

source /trdapps/linux-x86_64/envs/plis_conda/bin/activate /trdapps/linux-x86_64/envs/plis_conda/envs/synthseg_38
stdbuf -o0 python mongohead/worker.py
```

- Wait for the databases to saturate (this depends on your generator pipeline)
- Read from the database using MongoHeadDataset

```
from pymongo import MongoClient
from wirehead.MongoheadDataset import MongoHeadDataset

db = MongoClient(mongodbhostname)[database_name]
dataset = MongoheadDataset(collection = db[collection_name], sample = ('data', 'label'))
```
