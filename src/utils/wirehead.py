import redis
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import os
import pickle
import sys
import random
from datetime import datetime, timedelta

# Things that users should change
DEFAULT_HOST = 'arctrdagn019'
DEFAULT_PORT = 6379
DEFAULT_CAP = 5 
MANAGER_TIMEOUT = 1


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
       # "training_seg_05.nii.gz", 
        "training_seg_06.nii.gz",  "training_seg_07.nii.gz",  "training_seg_08.nii.gz",  "training_seg_09.nii.gz",  "training_seg_10.nii.gz",  "training_seg_11.nii.gz",  "training_seg_12.nii.gz",  "training_seg_13.nii.gz",  "training_seg_14.nii.gz",  "training_seg_15.nii.gz", "training_seg_16.nii.gz", "training_seg_17.nii.gz", "training_seg_18.nii.gz","training_seg_19.nii.gz", "training_seg_20.nii.gz",]


# Stuff for dataloader
MAX_RETRIES = 10
DATALOADER_SLEEP_TIME = 0.5

def get_queue_len(r):
    try:
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1

def quantize_to_uint8(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val == min_val:
        return np.zeros_like(tensor, dtype='uint8')
    tensor = ((tensor - min_val) / (max_val - min_val) * 255).round()
    return tensor.astype('uint8')

def lock_db(r, lock_name, timeout=10):
    while True:
        if r.setnx(lock_name, 1):
            r.expire(lock_name, timeout)
            return True
        time.sleep(0.1)

def swap_db(r):
    lock_name = 'swap_lock'
    locked = lock_db(r, lock_name = lock_name)
    if locked:
        try:
            pipe = r.pipeline()
            if r.exists("db0") and r.exists("db1"):
                pipe.multi()
                pipe.rename("db0", "temp_key")
                pipe.rename("db1", "db0")
                pipe.rename("temp_key", "db1")
                pipe.delete('db1')
                pipe.execute()
        finally:
            r.delete(lock_name)

def push_db(r, package_bytes):
    lock_name = 'swap_lock'
    locked = lock_db(r, lock_name = lock_name)
    if locked:
        try:
            r.rpush("db1", package_bytes)
            if not r.exists("db0"):
                r.rpush("db0", package_bytes)
        finally:
            r.delete(lock_name)

def load_fake_samples():
    im = np.load('/data/users1/mdoan4/wirehead/src/samples/image.npy')
    lab = np.load('/data/users1/mdoan4/wirehead/src/samples/label.npy')
    return im, lab

def hang_until_redis_is_loaded(r):
    while (True):
        try:
            r.rpush('status', bytes(True))
            break
            return
        except redis.ConnectionError:
            print(f"Redis is loading database...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Exiting.")
            break
            return
class Dataloader(Dataset):
    def __init__(self, transform, num_samples = int(1e6), fields=None, id="id", host=DEFAULT_HOST, port=DEFAULT_PORT):
        # Declare redis server to draw data from
        self.r = redis.Redis(host=host, port=port)
        # This whole block of code to check for redis status, and prevents script prematurely terminating if
        # redis is either not active or if the rotating database isn't filled up
        # Note: This dataloader will ALWAYS ATTEMPT to load, and will note terminate
        hang_until_redis_is_loaded(self.r)
        lendb0, lendb1 = get_queue_len(self.r)
        # Hangs while database is not ready
        while lendb0 < 2:
            print('Dataloader: Database is currently empty, please wait') 
            time.sleep(10)
            lendb0, lendb1 = get_queue_len(self.r)
        self.transform = transform
        self.db_key = 'db0'
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples 

    def __getitem__(self, index):
        r = self.r
        if not r.exists("wirehead_index"):
            r.set("wirehead_index", 0)
            index = 0
        while True:
            pickled_data = r.lindex(self.db_key, int(index))
            if pickled_data is not None:
                data = pickle.loads(pickled_data)
                r.incr("wirehead_index")
                index = int(r.get("wirehead_index"))
                if index > DEFAULT_CAP:
                    index = 0
                    r.set("wirehead_index", 0)
                return self.transform(data[0]), self.transform(data[1])
            else:
                time.sleep(DATALOADER_SLEEP_TIME)
                r.set("wirehead_index", 0)
                index = 0

#-- Utils ---------------------------
def time_between_calls():
    last_time = None
    while True:
        current_time = datetime.utcnow()
        if last_time:
            time_diff = current_time - last_time
            # Round to 4 decimal places
            time_diff = round(time_diff.total_seconds(), 4)
            yield f"{time_diff} seconds"
        last_time = current_time







