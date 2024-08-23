import sys
import time

import torch
from wirehead import MongoheadDataset

dataset = MongoheadDataset(config_path = "config.yaml")

for i in range(10000):
    idx = [0]
    data = dataset[idx]
    sample, label = data[0]['input'], data[0]['label']
    print("Loader, index: ", i)
    time.sleep(0.1)
