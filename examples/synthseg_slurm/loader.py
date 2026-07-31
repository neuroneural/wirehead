import sys
import time

import torch
from wirehead import WireheadDataset

dataset = WireheadDataset(config_path = "config.yaml")

for i in range(10000):
    idx = [0]
    data = dataset[idx]
    sample, label = data[idx[0]]['data'], data[idx[0]]['label']
    print("Loader, index: ", i)
    time.sleep(0.1)
