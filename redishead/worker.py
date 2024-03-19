import redis
import time
import os
import pickle
import sys
import random
import argparse
import numpy as np 
import torch
import io

sys.path.append('/data/users1/mdoan4/wirehead/dependencies/synthseg')
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

from wirehead_defaults import *

### Preprocessing related code
# Label map for converting labels from synthseg to contiguous labels
LABEL_MAP = np.asarray([ 0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 1, 16, 0, 17, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 19, 20, 21, 0, 22, 23, 0,
    24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab].astype(np.uint8)

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
    return img

def preprocess_image_min_max(img):
    "Min max scaling preprocessing for the range 0..1"
    img = ((img - img.min()) / (img.max() - img.min()))
    return img

### Redis related code
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
 
def tensor2bin(tensor):

    tensor = tensor.to(torch.uint8)

    # Serialize tensor and get binary
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    tensor_binary = buffer.getvalue()

    return tensor_binary

### Convenience and debugging tools
def measure_time(generation_time, generation_time_end, pickle_time, pickle_time_end, push_time, push_time_end):
    print(f"""
    {time.time()}
    ----------------------------------
    The pickling took {pickle_time_end - pickle_time}
    Pushing to the server took {push_time_end - push_time}
    The generation took in total {generation_time_end - generation_time}
    """)

def create_generator(training_seg=None):
    f"""
    This function is used every {GENERATOR_LENGTH} samples
    to refresh to the underlying ground truth used by
    SynthSeg"""
    if training_seg==None:
        training_seg = random.choice(DATA_FILES)
    else:
        training_seg = training_seg
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(f"Generator: SynthSeg is generating off {training_seg}")
    return brain_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    parser.add_argument("--training_seg", help="Segment to use for generation")
    parser.add_argument("--gen_len", help="Amount of samples to generate")

    args = parser.parse_args()

    host = args.ip if args.ip else DEFAULT_HOST
    port = args.port if args.port else DEFAULT_PORT
    training_seg = args.training_seg if args.training_seg else None
    gen_len = args.gen_len if args.gen_len else GENERATOR_LENGTH 
        
    r = connect_to_redis(host, port)
    #TODO: reimplement the rotating samples stuff # The better fix is managing this outside of the python script
    brain_generator = create_generator(training_seg)

    while(True):
        for i in range(gen_len):
            # Generate a sample using synthseg 
            total_time = time.time()
            generation_time = time.time()
            img, lab = brain_generator.generate_brain()
            generation_time_end = time.time()
            preprocess_time = time.time()

            # Preprocessing labels
            img = preprocess_image_min_max(img)*255 # Normalize and multiply by 255
            img = img.astype(np.uint8) # Quantize to uint8
            lab = preprocess_label(lab) # Convert non brain labels to 0 and map remaining labels corrently
            pickle_time = time.time()

            # Printing tensor info for sanity checks
            print("Img info:")
            get_tensor_info(img)
            print("Lab info:")
            get_tensor_info(lab)


            # Convert tensors to byte streams
            img = tensor2bin(torch.from_numpy(img))
            lab = tensor2bin(torch.from_numpy(lab))

            # Put the byte streamed tensors into a tuple
            package = (
                    img,
                    lab
                    )
            print(f'Preprocesing took {time.time()-preprocess_time}')

            # Convert the tuple into a byte stream 
            package_bytes = pickle.dumps(package)
            pickle_time_end = time.time()

            # Push the byte package to the server 
            push_redis(r, package_bytes)
            push_time = time.time()

            measure_time(
                    generation_time,
                    generation_time_end,
                    pickle_time,
                    pickle_time_end,
                    push_time,
                    time.time()
                    )

            print(f'In total, everything took {time.time()-total_time}')
