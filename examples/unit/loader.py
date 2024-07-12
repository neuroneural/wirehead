import torch
from wirehead import MongoheadDataset, MongoTupleheadDataset

dataset = MongoheadDataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx]
sample, label = data[0]['input'], data[0]['label']

dataset = MongoTupleheadDataset(config_path = "config.yaml")
idx = [0] 
data = dataset[idx][0]
sample, label = data[0], data[1]

print("Test passed")
