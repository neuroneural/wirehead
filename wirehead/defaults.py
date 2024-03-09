#######################
### Global defaults ###
#######################
MONGO_DBNAME    = 'wirehead_test'
MONGO_READ      = 'read'
MONGO_WRITE     = 'write'
CHUNKSIZE       = 10
SWAP_THRESHOLD  = 10000

###########################
### Defaults for TReNDS ###
###########################
MONGO_CLIENT    = 'mongodb://10.245.12.58:27017/' # arctrdgn018
WIREHEAD_PATH   = "/data/users1/mdoan4/wirehead/"
SYNTHSEG_PATH   = '/data/users1/mdoan4/wirehead/dependencies/synthseg'
DATA_PATA       = SYNTHSEG_PATH + "/data/training_label_maps/"
DATA_FILES      = [ "training_seg_01.nii.gz","training_seg_02.nii.gz","training_seg_03.nii.gz",
                    "training_seg_04.nii.gz","training_seg_05.nii.gz","training_seg_06.nii.gz",
                    "training_seg_07.nii.gz","training_seg_08.nii.gz","training_seg_09.nii.gz",
                    "training_seg_10.nii.gz","training_seg_11.nii.gz","training_seg_12.nii.gz",
                    "training_seg_13.nii.gz","training_seg_14.nii.gz","training_seg_15.nii.gz",
                    "training_seg_16.nii.gz","training_seg_17.nii.gz","training_seg_18.nii.gz",
                    "training_seg_19.nii.gz","training_seg_20.nii.gz"]


