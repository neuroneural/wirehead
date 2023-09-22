import redis
import pickle
import sys
import torch
from time import time

# Connect to your Redis instance.
r = redis.Redis(host='localhost', port=6379, db=0)

# Extracting a single sample from the cache to get metrics
start_lpop = time()
im_bytes= r.lpop('mylist') # returns 'item1'
start_deserializing = time()
im = pickle.loads(im_bytes)
dtype = im.dtype
print(f"Deserializing took {time()-start_deserializing}")
print(f"Loading in total took {time()-start_lpop}")

# Calulate metrics for redis database (type, size, number of samples)
info = r.info()
im_size_in_bytes = sys.getsizeof(im_bytes)
memory_used = info['used_memory']
num_samples = memory_used/im_size_in_bytes
print(f"Numbers are stored in {dtype} type")
print(f"Synthetic data size: {im_size_in_bytes} bytes")
print(f"Memory used: {memory_used} bytes")
print(f"Number of samples in cache: {num_samples}")



