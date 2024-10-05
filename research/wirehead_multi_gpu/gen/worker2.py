import os
import sys
import numpy as np
import csv
import time
from datetime import datetime
import argparse
from wirehead import WireheadManager, WireheadGenerator

# Synthseg config
WIREHEAD_CONFIG     = "./conf/wirehead_config.yaml"
PATH_TO_DATA        = ("./gen/SynthSeg/data/training_label_maps/")
DATA_FILES          = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
PATH_TO_SYNTHSEG    = './gen/SynthSeg/'

LABEL_MAP = np.asarray(
    [0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10]
    + [11, 12, 13, 14, 15]
    + [0] * 6
    + [1, 16, 0, 17]
    + [0] * 12
    + [18, 19, 20, 21, 0, 22, 23]
    + [0, 24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31]
    + [0] * 75
    + [3, 4]
    + [0] * 25
    + [20, 21]
    + [0] * 366,
    dtype="int",
).astype(np.uint8)

def merge_homologs(label):#, device):
    
    max_value = 105
    print("this is me, in merge_homologs")
    # idx = torch.arange(max_value + 1, dtype=torch.long).to(device)
    idx = np.arange(max_value+1)
    print("this is me after idx has been declared")
    idx[[2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]] = 1
    idx[[26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]] = 1
    
    #csf
    idx[24] = 1
    
    idx[[0, 100, 101, 102, 103, 104, 105]] = 0
    # return the corresponding values from idx
    print("this is me, after labels have been assigned")
    return idx[label]

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab.astype(np.uint8)]

def preprocess_image_min_max(img: np.ndarray):
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img

def preprocessing_pipe(data):
    """ Set up your preprocessing options here, ignore if none are needed """
    img, lab = data
    print("Img is being preprocessed")
    img = preprocess_image_min_max(img) * 255
    print("Img type is being converted to uint8")
    img = img.astype(np.uint8)
    print("Lab is being preprocessed")
    lab = preprocess_label(lab)
    print("Lab is getting pushed through merge_homologs")
    lab = merge_homologs(lab)
    print("Lab is being converted to uint8")
    lab = lab.astype(np.uint8)    
    print("Lab has been converted")
    return (img, lab) 



def hardware_setup():
    """ Clean slate to set up your hardware, ignore if none are needed """
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    sys.path.append(PATH_TO_SYNTHSEG)
    pass

# Create a generator function that yields desired samples
def create_generator(task_id, training_seg=None):
    """ Creates an iterator that returns data for mongo.
        Should contain all the dependencies of the brain generator
        Preprocessing should be applied at this phase 
        yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str)) """

    # 0. Optionally set up hardware configs
    hardware_setup()
    # 1. Declare your generator and its dependencies here
    sys.path.append(PATH_TO_SYNTHSEG)
    from SynthSeg.brain_generator import BrainGenerator
    training_seg = DATA_FILES[task_id % len(DATA_FILES)] if training_seg == None else training_seg
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(f"Generator: SynthSeg is generating off {training_seg}",flush=True,)
    # 2. Run your generator in a loop, and pass in your preprocessing options
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        print("preprocessing has been completed in generator")
        # 3. Yield your data, which will automatically be pushed to mongo
        #np.save('img.npy', img.cpu().numpy())
        #print("img has been saved as npy")
        #np.save('lab.npy', lab.cpu().numpy())
        #print("lab has been saved as npy")


        try:
            # Ensure img and lab are on CPU and converted to numpy arrays if they are tensors
            np.save('img.npy', img)
            print("img has been saved as npy")
            np.save('lab.npy', lab)
            print("lab has been saved as npy")
        except Exception as e:
            print(f"Error saving npy files: {e}")

        yield (img, lab)

# Extras
def my_task_id():
    """ Returns slurm task id """
    task_id = os.getenv(
        "SLURM_ARRAY_TASK_ID", "0"
    )  # Default to '0' if not running under Slurm
    return int(task_id)

# Function to check if this is the first job based on SLURM_ARRAY_TASK_ID
def is_first_job():
    return my_task_id() == 0

if __name__ == "__main__":
    brain_generator    = create_generator(my_task_id())
    wirehead_generator = WireheadGenerator(generator = brain_generator, config_path = WIREHEAD_CONFIG)
    wirehead_generator.run_generator()
