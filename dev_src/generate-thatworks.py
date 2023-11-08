import redis
import time
import os
import pickle
import sys
import random
import argparse

sys.path.append('/data/users1/mdoan4/wirehead/synthseg')
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

DEFAULT_HOST = 'localhost'
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


def lock_db(r, lock_name, timeout=10):
    while True:
        if r.setnx(lock_name, 1):
            r.expire(lock_name, timeout)
            return True
        time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    args = parser.parse_args()

    host = args.ip if args.ip else DEFAULT_HOST
    port = args.port if args.port else DEFAULT_PORT

    training_seg = random.choice(DATA_FILES)
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(training_seg)
    r = redis.Redis(host=host, port = port)

    while(True):
        start_time = time.time()
        im, lab = brain_generator.generate_brain()
        pickle_time = time.time()
        package = (im,lab)
        package_bytes = pickle.dumps(package)
        print(f"The pickling took {time.time() - pickle_time}")

        # Push to db1
        print(f"The generation took {time.time() - start_time}")
        lock_name = 'swap_lock'
        locked = lock_db(r, lock_name = lock_name)
        if locked:
            try:
                r.rpush("db1", package_bytes)
                if not r.exists("db0"):
                    r.rpush("db0", package_bytes)
            finally:
                r.delete(lock_name)


        print('----------------------------------')

