import redis
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import os
import pickle
import sys
import random
from datetime import datetime, timedelta
from wirehead_defaults import *
from multiprocessing import Lock

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

class whDataset(Dataset):
    def __init__(self, transform, num_samples=int(1e6), host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.transform = transform
        self.db_key = 'db0'
        self.num_samples = num_samples
        self.host=host
        self.port=port
        r = redis.Redis(host=self.host, port=self.port)
        hang_until_redis_is_loaded(r)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        r = redis.Redis(host=self.host, port=self.port)
        index = index % r.llen(self.db_key)  # Use modular arithmetic to cycle through dataset
        pickled_data = r.lindex(self.db_key, index)
        if pickled_data is not None:
            data = pickle.loads(pickled_data)
            return self.transform(data[0]), self.transform(data[1])
        else:
            raise IndexError(f"Index {index} out of range")


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

class Dataloader_for_tests(Dataset):
    def __init__(self, host, port, transform, num_samples):
        self.host = host
        self.port = port
        self.transform = transform
        self.num_samples = num_samples
        #self.lock = Lock()
    def __len__(self):
        #with self.lock:
        return self.num_samples 

    def __getitem__(self, index):
        #with self.lock:
        time.sleep(1) 
        tensor1 = np.zeros((256, 256, 256), dtype=np.float32)
        tensor2 = np.zeros((256, 256, 256), dtype=np.uint8)
        return self.transform(tensor1), self.transform(tensor2)

