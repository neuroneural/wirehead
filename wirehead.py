import redis
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import os
import pickle
import sys
import random

# Things that users should change
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 6379
DEFAULT_CAP = 1000
MANAGER_TIMEOUT = 1


# Things that users should definitely NOT change
ERROR_STRING= """Oppsie Woopsie! Uwu Redwis made a shwuky wucky!! A widdle
bwucko boingo! The code monkeys at our headquarters are
working VEWY HAWD to fix this!"""
PATH_TO_DATA = "/data/users1/mdoan4/wirehead/synthseg/data/training_label_maps/"
DATA_FILES = ["training_seg_01.nii.gz",  "training_seg_02.nii.gz",  "training_seg_03.nii.gz",  "training_seg_04.nii.gz",  "training_seg_05.nii.gz",  "training_seg_06.nii.gz",  "training_seg_07.nii.gz",  "training_seg_08.nii.gz",  "training_seg_09.nii.gz",  "training_seg_10.nii.gz",  "training_seg_11.nii.gz",  "training_seg_12.nii.gz",  "training_seg_13.nii.gz",  "training_seg_14.nii.gz",  "training_seg_15.nii.gz", "training_seg_16.nii.gz", "training_seg_17.nii.gz", "training_seg_18.nii.gz","training_seg_19.nii.gz", "training_seg_20.nii.gz",]


# Stuff for dataloader
MAX_RETRIES = 10
DATALOADER_SLEEP_TIME = 0.5

def get_queue_len(r):
    try:
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1

def quantize_to_uint8(tensor):
    tensor = ((tensor - tensor.min())/(tensor.max() - tensor.min())*255).round() 
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

class wirehead_dataloader(Dataset):
    def __init__(self, host = DEFAULT_HOST, port = DEFAULT_PORT):
        self.r = redis.Redis(host=host, port=port)
        hang_until_redis_is_loaded(self.r)
        lendb0, lendb1 = get_queue_len(self.r)
        while lendb0 < 2:
            print('Dataloader: Database is currently empty, please wait')
            time.sleep(10)
            lendb0, lendb1 = get_queue_len(self.r)
        self.db_key = 'db0'
    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        while True:
            pickled_data = self.r.lpop(self.db_key)
            if pickled_data is not None:
                self.r.rpush(self.db_key, pickled_data)
                data = pickle.loads(pickled_data)
                return data[0], data[1]
            else:
                time.sleep(0.5)
            return [0,0]

class MongoBatchDataset(Dataset):
    def __init__(self, indices, transform, collection, fields=None, id="id"):
        self.indices = indices
        self.transform = transform
        self.collection = collection
        self.fields = fields
        self.id = id
    def __len__(self):
        return int(1e6)
    def __getitem__(self, batch):
        if self.fields is None:
            field_list = {}
        else:
            field_list = {_: 1 for _ in self.fields}
        samples = self.collection.find(
            {self.id: {"$in": [self.indices[_] for _ in batch]}}, field_list
        )
        return [self.transform(_) for _ in samples]


class wirehead_dataloader_v1(Dataset):
    def __init__(self, transform, fields=None, id="id", host=DEFAULT_HOST, port=DEFAULT_PORT):
        # Declare redis server to draw data from
        self.r = redis.Redis(host=host, port=port)
        # This whole block of code to check for redis status, and prevents script prematurely terminating if
        # redis is either not active or if the rotating database isn't filled up
        # Note: This dataloader will ALWAYS ATTEMPT to load, and will note terminate
        hang_until_redis_is_loaded(self.r)
        lendb0, lendb1 = get_queue_len(self.r)
        while lendb0 < 2:
            print('Dataloader: Database is currently empty, please wait') # For debugging, remove in prod
            time.sleep(10)
            lendb0, lendb1 = get_queue_len(self.r)
        # Actual inits are here
        self.transform = transform
        self.db_key = 'db0'

    def __len__(self):
        return int(1e6)

    # todo:
    # - Pushing and popping to redis is a bottle neck, need to fix
    # - Manually unpickling the database is a hassle, need to fix
    #   - Maybe instead of loading from database I move to a local cache with db0? 
    #     This would greatly simplify code and also removes overhead of manually
    #     processing all elements every time a load is required
    # - Pickling is also an overhead, need to replace
    def __getitem__(self, index):
        # Loop is just in case redis is not reachable
        retries = 0
        while retries < MAX_RETRIES:
            pickled_data = self.r.lpop(self.db_key)
            if pickled_data is not None:
                self.r.rpush(self.db_key, pickled_data)
                data = pickle.loads(pickled_data)
                return self.transform(data[0]), self.transform(data[1])
            else:
                time.sleep(DATALOADER_SLEEP_TIME)
                retries += 1
        raise Exception("Dataloader: Max retries reached, database unavailable.")







