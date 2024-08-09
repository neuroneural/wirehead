import sys

import torch
from wirehead import mongoheaddataset

dataset = mongoheaddataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx]
sample, label = data[0]['input'], data[0]['label']
print(sample.shape)
print(label.shape)
print("fetched successfully")
if sample.shape == label.shape and sample.shape == (256,256,256):
    print("test passed")
