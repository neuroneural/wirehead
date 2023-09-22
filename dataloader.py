import torch
from torch.utils.data import DataLoader, Dataset
from time import time
import redis
import numpy as np
import pickle
import sys

max_fetch_len = 4    
batch_size = 2

host = 'localhost'
queue_name = 'mylist'
redis_port = 6379

class RedisQueueDataset(Dataset):
    def __init__(self, redis_host=host, redis_port=redis_port, queue_name=queue_name):
        self.redis_conn = redis.Redis(host=redis_host, port=redis_port)
        self.queue_name = queue_name
    def __len__(self):
        return min(max_fetch_len, self.redis_conn.llen(self.queue_name))
    def __getitem__(self, index):
        data = self.redis_conn.lindex(self.queue_name, index)
        if data is None:
            return None
        ndarray = pickle.loads(data)
        return torch.tensor(ndarray)

def lpop(r, queue_name, sample_count):
    for i in range(sample_count):
        im_bytes = r.lpop(queue_name)
    im_bytes= r.lpop('mylist') # returns 'item1'
    start_deserializing = time()
    im = pickle.loads(im_bytes)
    dtype = im.dtype
    info = r.info()
    im_size_in_bytes = sys.getsizeof(im_bytes)
    memory_used = info['used_memory']
    num_samples = memory_used/im_size_in_bytes
    print(f"Numbers are stored in {dtype} type")
    print(f"Synthetic data size: {im_size_in_bytes} bytes")
    print(f"Memory used: {memory_used} bytes")
    print(f"Number of samples in cache: {num_samples}")

    
if __name__ == "__main__":
    dataset = RedisQueueDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample_count = 0
    for i, batch in enumerate(dataloader):
        print(batch)
        sample_count += i*batch_size
        # Do stuff with the data loader
    
    # Only if the script terminates properly does the total number of samples used
    # by the dataloader gets popped from the queue 
    r = redis.Redis(host=host, port=redis_port, db=0)
    lpop(r, queue_name, sample_count)
    
