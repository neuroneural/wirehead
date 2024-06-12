import sys

import torch
sys.path.append('/data/users1/mdoan4/wirehead')
from wirehead import MongoheadDataset

dataset = MongoheadDataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx]
sample, label = data[0]['input'], data[0]['label']
print(sample.shape)
print(label.shape)
print("Fetched successfully")
