#######################
### Global defaults ###
#######################
SWAP_THRESHOLD  = 10000
MONGO_DBNAME    = 'wirehead'
MONGO_READ      = 'read'
MONGO_WRITE     = 'write'

#######################################################
### Defaults for TReNDS, change to match your setup ###
#######################################################
MONGO_CLIENT    = 'mongodb://10.245.12.58:27017/' # arctrdgn018
WIREHEAD_PATH   = "/data/users1/mdoan4/wirehead/"
DATA_PATA       = PATH_TO_WIREHEAD + "dependencies/synthseg/data/training_label_maps/"
DATA_FILES      = [ "training_seg_01.nii.gz","training_seg_02.nii.gz","training_seg_03.nii.gz",
                    "training_seg_04.nii.gz","training_seg_05.nii.gz","training_seg_06.nii.gz",
                    "training_seg_07.nii.gz","training_seg_08.nii.gz","training_seg_09.nii.gz",
                    "training_seg_10.nii.gz","training_seg_11.nii.gz","training_seg_12.nii.gz",
                    "training_seg_13.nii.gz","training_seg_14.nii.gz","training_seg_15.nii.gz",
                    "training_seg_16.nii.gz","training_seg_17.nii.gz","training_seg_18.nii.gz",
                    "training_seg_19.nii.gz","training_seg_20.nii.gz"]

