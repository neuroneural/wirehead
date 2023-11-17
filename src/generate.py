import redis
import time
import os
import pickle
import sys
import random
import argparse
import numpy as np 

sys.path.append('/data/users1/mdoan4/wirehead/synthseg')
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

from wirehead_defaults import *

# Redis connection and queue handling functions
def get_queue_len(host = DEFAULT_HOST, port = DEFAULT_PORT):
    try:
        r = redis.Redis(host= DEFAULT_HOST, port=DEFAULT_PORT, db=0)
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1
def quantize_to_uint8(tensor):
    tensor = ((tensor - tensor.min())/(tensor.max() - tensor.min*())*255).round()#Normalize
    return tensor.astype('uint8') # convert to uint8

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
 

# Sample preprocessing functions
def preprocess_label(lab):
    "Convert unique labels into range [0..52]"
    label_to_int = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 24: 16, 25: 17, 26: 18, 28: 19, 30: 20, 41: 21, 42: 22, 43: 23, 44: 24, 46: 25, 47: 26, 49: 27, 50: 28, 51: 29, 52: 30, 53: 31, 54: 32, 57: 33, 58: 34, 60: 35, 62: 36, 72: 37, 85: 38, 136: 39, 137: 40, 163: 41, 164: 42, 502: 0, 506: 0, 507: 0, 508: 0, 509: 0, 511: 0, 512: 0, 514: 0, 515: 0, 516: 0, 530: 0}
    # Vectorized mapping of original labels to contiguous range
    vectorized_map = np.vectorize(lambda x: label_to_int.get(x, -1))  # -1 as default for unmapped labels
    lab = vectorized_map(lab)
    print(np.unique(lab))
    return lab.astype(np.uint8)

def preprocess_image(img, qmin=0.01, qmax=0.99):
    """Unit interval preprocessing"""
    qmin_value = np.quantile(img, qmin)
    qmax_value = np.quantile(img, qmax)
    img = (img - qmin_value) / (qmax_value - qmin_value)
    return img

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

    # Main generator loop
    while(True):
        brain_generator = create_generator()
        for i in range(GENERATOR_LENGTH):
            # Start of generation
            generation_time = time.time()
            img, lab = brain_generator.generate_brain()
            generation_time_end = time.time()
            # Start of pickling
            pickle_time = time.time()

            package = (
                    preprocess_image(img),
                    preprocess_label(lab)
                    )

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

