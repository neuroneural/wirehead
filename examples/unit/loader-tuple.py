import torch
from wirehead import MongoTupleheadDataset

dataset = MongoTupleheadDataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx][0]
sample, label = data[0], data[1]
print(sample.shape)
print(label.shape)
print("Fetched successfully")
