import torch
from torch.utils.data import DataLoader, Dataset
import time
import redis
import numpy as np
import pickle
import wirehead as wh


if __name__ == "__main__":
    dataset = wh.wirehead_dataloader()
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        im, lab = batch[0], batch[1]
        print("Dataloader: Fetched im and lab successfully")

        
  
