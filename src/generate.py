import redis
import time
import os
import pickle
import sys
import random
import argparse
import numpy as np 

sys.path.append('/data/users1/mdoan4/wirehead/dependencies/synthseg')
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

from wirehead_defaults import *

# Redis connection and queue handling functions
LABEL_MAP = np.asarray([ 0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 1, 16, 0, 17, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 19, 20, 21, 0, 22, 23, 0,
    24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0], dtype='int').astype(np.uint8)

def push_redis(r, package_bytes):
    def lock_db(r, lock_name, timeout=10):
        """
        This function ensures proper concurrency mangagement
        by redis. Unsafe to edit without extensive testing
        """
        while True:
            if r.setnx(lock_name, 1):
                r.expire(lock_name, timeout)
                return True
            time.sleep(0.1)
    lock_name = 'swap_lock'
    locked = lock_db(r, lock_name = lock_name)
    if locked:
        try:
            r.rpush("db1", package_bytes)
            if not r.exists("db0"):
                r.rpush("db0", package_bytes)
        finally:
            r.delete(lock_name)

def connect_to_redis(host, port):
    def hang_until_redis_is_loaded(r):
        while (True):
            try:
                r.rpush('status', bytes(True))
                break
                return 
            except redis.ConnectionError:
                print(f"Generator: Redis is not responding") 
                time.sleep(5)
            except KeyboardInterrupt:
                print("Generator: Terminating at Redis loading.")
                break
                return None
    while(True):
        try:
            r = redis.Redis(host=host, port = port)
            hang_until_redis_is_loaded(r)
            print(f"Generator: Connected to Redis hosted at {host}:{port}")
            return r
        except redis.ConnectionError:
            print(f"Generator: Redis is not responding") 
            time.sleep(5)
        except KeyboardInterrupt:
            print("Generator: Terminating at Redis loading.")
            break
            return None 
 

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab.astype(np.uint8)]

def get_tensor_info(tensor):
    min_value = np.min(tensor)
    max_value = np.max(tensor)
    shape = tensor.shape
    dtype = tensor.dtype

    print(f"Min Value : {min_value}")
    print(f"Max Value : {max_value}")
    print(f"Shape     : {shape}")
    print(f"Data Type : {dtype}")

def preprocess_image_quantile(img, qmin=0.01, qmax=0.99):
    "Unit interval preprocessing for quantile normalization"
    qmin_value = np.quantile(img, qmin)
    qmax_value = np.quantile(img, qmax)
    img = (img - qmin_value) / (qmax_value - qmin_value)
    return img.astype(np.uint8)

def preprocess_image_min_max(img):
    "Min max scaling preprocessing for the range 0..1"
    img = ((img - img.min()) / (img.max() - img.min()))
    return img.astype(np.uint8)


def measure_time(generation_time, generation_time_end, pickle_time, pickle_time_end, push_time, push_time_end):
    print(f"""
    {time.time()}
    ----------------------------------
    The pickling took {pickle_time_end - pickle_time}
    Pushing to the server took {push_time_end - push_time}
    The generation took in total {generation_time_end - generation_time}
    """)

def create_generator():
    f"""
    This function is used every {GENERATOR_LENGTH} samples
    to refresh to the underlying ground truth used by
    SynthSeg"""
    training_seg = random.choice(DATA_FILES)
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(f"Generator: SynthSeg is generating off {training_seg}")
    return brain_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    args = parser.parse_args()

    host = args.ip if args.ip else DEFAULT_HOST
    port = args.port if args.port else DEFAULT_PORT

    r = connect_to_redis(host, port)
    #TODO: reimplement the rotating samples stuff
    brain_generator = create_generator()
    # Main generator loop
    while(True):
        for i in range(GENERATOR_LENGTH):
            # Start of generation
            total_time = time.time()
            generation_time = time.time()
            img, lab = brain_generator.generate_brain()
            generation_time_end = time.time()

            # Start of pickling
            pickle_time = time.time()
            preprocess_time = time.time()
            img = preprocess_image_min_max(img)*255
            lab = preprocess_label(lab)
            print("Img info:")
            get_tensor_info(img)
            print("Lab info:")
            get_tensor_info(lab)
            package = (
                    img,
                    lab
                    )
            print(f'Preprocesing took {time.time()-preprocess_time}')
            package_bytes = pickle.dumps(package)
            pickle_time_end = time.time()
            # Start of pushing to server
            push_time = time.time()
            push_redis(r, package_bytes)
            measure_time(
                    generation_time,
                    generation_time_end,
                    pickle_time,
                    pickle_time_end,
                    push_time,
                    time.time()
                    )

            print(f'In total, everything took {time.time()-total_time}')
