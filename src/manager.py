import redis
import time
import os
import pickle
import sys
import random

# Things that users should change
DEFAULT_HOST = 'arctrdagn019'
DEFAULT_PORT = 6379
DEFAULT_CAP = 30 
MANAGER_TIME_INTERVAL = 5


# Things that users should definitely NOT change
ERROR_STRING= """Oppsie Woopsie! Uwu Redwis made a shwuky wucky!! A widdle
bwucko boingo! The code monkeys at our headquarters are
working VEWY HAWD to fix this!"""
PATH_TO_DATA = "/data/users1/mdoan4/wirehead/synthseg/data/training_label_maps/"
DATA_FILES = ["training_seg_01.nii.gz",  "training_seg_02.nii.gz",  "training_seg_03.nii.gz",  "training_seg_04.nii.gz",  "training_seg_05.nii.gz",  "training_seg_06.nii.gz",  "training_seg_07.nii.gz",  "training_seg_08.nii.gz",  "training_seg_09.nii.gz",  "training_seg_10.nii.gz",  "training_seg_11.nii.gz",  "training_seg_12.nii.gz",  "training_seg_13.nii.gz",  "training_seg_14.nii.gz",  "training_seg_15.nii.gz", "training_seg_16.nii.gz", "training_seg_17.nii.gz", "training_seg_18.nii.gz","training_seg_19.nii.gz", "training_seg_20.nii.gz",]
LOCK_NAME = "pipeline_lock"

def get_queue_len(host = DEFAULT_HOST, port = DEFAULT_PORT):
    try:
        r = redis.Redis(host= DEFAULT_HOST, port=DEFAULT_PORT, db=0)
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1
def quantize_to_uint8(tensor):
    tensor = ((tensor - tensor.min())/(tensor.max() - tensor.min())*255).round()#Normalize
    return tensor.astype('uint8') # convert to uint8

def acquire_lock(r, lock_name=LOCK_NAME, timeout=10):
    while True:
        if r.setnx(lock_name, 1):
            r.expire(lock_name, timeout)
            return True
        time.sleep(0.1)

def release_lock(r, lock_name=LOCK_NAME):
    r.delete(lock_name)

# Usage in your code


if __name__ == '__main__':
    cap = DEFAULT_CAP 
    r = redis.Redis(host=DEFAULT_HOST, port = DEFAULT_PORT)

    while(True):
        try:
            lendb0, lendb1 = get_queue_len()
            print(lendb0, lendb1)
            if lendb0 == -1:
                print("db1 kinda doesn't exist")
                while lendb0 == -1:
                    time.sleep(5)
                    lendb0, lendb1 = get_queue_len()
                    continue
            # Swap databases whenever db1 is full
            if lendb1 >= cap:
                pipe = r.pipeline()
                """
                if not pipe.exists('db0'):
                    package_bytes = bytes(0)
                    pipe.rpush('db0', package_bytes)
                """
                pipe.rename("db0", "temp_key")
                pipe.rename("db1", "db0")
                pipe.rename("temp_key", "db1")
                #pipe.delete('db1')
                pipe.execute()
            # Timeout between checks 
            time.sleep(MANAGER_TIME_INTERVAL)
        except:
            print(ERROR_STRING)
            continue

