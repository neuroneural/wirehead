import torch
from torch.utils.data import DataLoader, Dataset
import time
import redis
import numpy as np
import pickle
import wirehead as wh

def identity(sample):
    return sample

if __name__ == "__main__":
    dataset = wh.wirehead_dataloader_v2(transform=identity)
    dataloader = DataLoader(dataset, batch_size=1)
    
    total_time = 0.0
    num_samples = 0
    
    for batch in dataloader:
        start_time = time.time()
        
        im, lab = batch[0], batch[1]
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 60
        
        total_time += elapsed_time
        num_samples += 1
        avg_time = total_time / num_samples 
        
        samples_per_sec = 1.0 / avg_time
        print(im)
        

