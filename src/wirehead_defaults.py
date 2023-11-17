if __name__ == '__main__':
    print("""
This file exists because not all the defaults can be 
imported safely from wirehead.py. There are versioning
conflicts due to SynthSeg and the training environments
using different versions of python that require different
things. So before I figure out how to solve this mess,
this is the solution that we'll have to work with 
unfortunately
""")

# Things that users should change
DEFAULT_HOST = 'arctrdagn019'
DEFAULT_PORT = 6379

# Things that users should definitely NOT change
ERROR_STRING= """Oppsie Woopsie! Uwu Redwis made a shwuky wucky!! A widdle
bwucko boingo! The code monkeys at our headquarters are
working VEWY HAWD to fix this!"""
PATH_TO_DATA = "/data/users1/mdoan4/wirehead/synthseg/data/training_label_maps/"
DATA_FILES = [
        "training_seg_01.nii.gz",  
        "training_seg_02.nii.gz",  
        "training_seg_03.nii.gz",  
        "training_seg_04.nii.gz",  
        "training_seg_05.nii.gz", 
        "training_seg_06.nii.gz",  "training_seg_07.nii.gz",  "training_seg_08.nii.gz",  "training_seg_09.nii.gz",  "training_seg_10.nii.gz",  "training_seg_11.nii.gz",  "training_seg_12.nii.gz",  "training_seg_13.nii.gz",  "training_seg_14.nii.gz",  "training_seg_15.nii.gz", "training_seg_16.nii.gz", "training_seg_17.nii.gz", "training_seg_18.nii.gz","training_seg_19.nii.gz", "training_seg_20.nii.gz",] # this is ugly, need to fix later

# Stuff for dataloader
MAX_RETRIES = 10
DATALOADER_SLEEP_TIME = 0.5

# Stuff for generator
GENERATOR_LENGTH = 100

# Stuff for manager
MANAGER_TIMEOUT = 1
DEFAULT_CAP = 500 # When to rotate the databases 


