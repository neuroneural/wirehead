import sys

import torch
from wirehead import WireheadDataset

dataset = WireheadDataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx]
sample, label = data[idx[0]]['data'], data[idx[0]]['label']
print(sample.shape)
print(label.shape)
print("Fetched successfully")
if sample.shape == label.shape and sample.shape == (256,256,256):
    print("Test passed")
