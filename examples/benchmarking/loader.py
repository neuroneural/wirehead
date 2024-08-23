import sys
import time

import torch
from wirehead import MongoTupleheadDataset

dataset = MongoTupleheadDataset(config_path = "config.yaml")

for i in range(1000000):
    idx = [i%20]
    data = dataset[idx]
    sample, label = data[0][0], data[0][1]
    print("Loader, index: ", idx)

print("Unit example passed successfully")
