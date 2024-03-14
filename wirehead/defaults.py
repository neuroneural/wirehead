import torch
#######################
### Global defaults ###
#######################
MONGO_DBNAME    = 'wirehead_test'
MONGO_READ      = 'read'
MONGO_WRITE     = 'write'
CHUNKSIZE       = 10
NUMCHUNKS       = 2
SWAP_THRESHOLD  = 100

###########################
### Defaults for TReNDS ###
###########################
MONGO_CLIENT    = 'mongodb://arctrdcn018:27017/' # 10.245.12.58
WIREHEAD_PATH   = "/data/users1/mdoan4/wirehead/"
SYNTHSEG_PATH   = '/data/users1/mdoan4/wirehead/dependencies/synthseg'
DATA_PATH       = SYNTHSEG_PATH + "/data/training_label_maps/"
DATA_FILES      = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
### These are for exception handling ### Please help figure out a better solution
DEFAULT_IMG     = torch.randint(0, 10, (256, 256, 256), dtype=torch.uint8)
DEFAULT_LAB     = torch.randint(0, 10, (256, 256, 256), dtype=torch.uint8) 