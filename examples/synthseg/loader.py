import sys

import torch
from wirehead import MongoheadDataset

dataset = MongoheadDataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx]
sample, label = data[0]['input'], data[0]['label']
print(sample.shape)
print(label.shape)
print("Fetched successfully")
if sample.shape == label.shape and sample.shape == (256,256,256):
    print("Test passed")
