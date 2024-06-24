import os
from datetime import datetime
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Custom dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n = 128
        shape = (n,n,n)
        input_data = torch.rand(*shape, dtype=torch.float32)
        label_data = torch.randint(0, 2, shape, dtype=torch.int32)
        return input_data, label_data

