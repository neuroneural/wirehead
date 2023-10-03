import redis
import time
import os
import pickle
import sys
import random

sys.path.append('/data/users1/mdoan4/wirehead/synthseg')
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

DEFAULT_HOST = 'arctrdagn019'
DEFAULT_PORT = 6379
error_string = """Oppsie Woopsie! Uwu Redwis made a shwuky wucky!! A widdle
bwucko boingo! The code monkeys at our headquarters are
working VEWY HAWD to fix this!"""
PATH_TO_DATA = "/data/users1/mdoan4/wirehead/synthseg/data/training_label_maps/"
DATA_FILES = ["training_seg_01.nii.gz",  "training_seg_02.nii.gz",  "training_seg_03.nii.gz",  "training_seg_04.nii.gz",  "training_seg_05.nii.gz",  "training_seg_06.nii.gz",  "training_seg_07.nii.gz",  "training_seg_08.nii.gz",  "training_seg_09.nii.gz",  "training_seg_10.nii.gz",  "training_seg_11.nii.gz",  "training_seg_12.nii.gz",  "training_seg_13.nii.gz",  "training_seg_14.nii.gz",  "training_seg_15.nii.gz", "training_seg_16.nii.gz", "training_seg_17.nii.gz", "training_seg_18.nii.gz","training_seg_19.nii.gz", "training_seg_20.nii.gz",]

def get_queue_len(host = DEFAULT_HOST, port = DEFAULT_PORT):
    try:
        r = redis.Redis(host= DEFAULT_HOST, port=DEFAULT_PORT, db=0)
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1
def quantize_to_uint8(tensor):
    tensor = ((tensor - tensor.min())/(tensor.max() - tensor.min*())*255).round()#Normalize
    return tensor.astype('uint8') # convert to uint8

if __name__ == '__main__':
    cap = 100
    r = redis.Redis(host=DEFAULT_HOST, port = DEFAULT_PORT)
    training_seg = random.choice(DATA_FILES)
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(training_seg)

    while(True):
        lendb0, lendb1 = get_queue_len()
        if lendb0 == -1:
            print(error_string)
            while lendb0 == -1:
                time.sleep(5)
                lendb0, lendb1 = get_queue_len()
                continue
        print(lendb0, lendb1)
        start_time = time.time()
        im, lab = brain_generator.generate_brain()
        #random_number = random.randint(1, 10000)
        #im, lab = random_number,1
        pickle_time = time.time()
        package = (im,lab)
        package_bytes = pickle.dumps(package)
        print(f"The pickling took {time.time() - pickle_time}")

        # Push to db1
        r.rpush("db1", package_bytes)
        print(f"The generation took {time.time() - start_time}")
        print('----------------------------------')
        time.sleep(1)

