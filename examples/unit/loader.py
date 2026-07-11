import sys
import time

import torch
from wirehead import WireheadDataset

dataset = WireheadDataset(config_path = "config.yaml")

for i in range(100):
    idx = [i%10]
    data = dataset[idx]
    sample, label = data[idx[0]]["data"], data[idx[0]]["label"]
    print("Loader, index: ", i)

print("Unit example passed successfully")
