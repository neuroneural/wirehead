import os 
from datetime import datetime
import shutil
import sys
import csv
import time
import subprocess
import threading

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.model import UNet
from utils.dice import DiceLoss
from utils.logging import Logger, gpu_monitor 
from utils.misc import RandomDataset
from utils.generator import SynthsegDataset

from wirehead import MongoTupleheadDataset

### Userland ###
use_wandb = False
# wandb_project = "wirehead_1x3090_baseline"
wandb_project = "wirehead_1x3090_wirehead"


# Hyperparameters
batch_size = 1         # this should be 1 to match synthseg
learning_rate = 1e-4   # this should be 1 to match synthseg
n_channels = 1         # unclear
n_classes = 2          # unclear 
num_samples = 20
num_epochs = 50        # 50 * 20 = 1000
assert num_samples*num_epochs == 1000, "total samples read should be 1000"
num_generators = 1     # unclear
dtype = torch.bfloat16  

### outside ###

# Logging constants
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_dir = f"./log/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
model_path          = os.path.join(log_dir, "unet_model.pth")
train_script_path   = os.path.join(log_dir, "train.py")
output_path         = os.path.join(log_dir, "output.txt")
gpu_csv_path        = os.path.join(log_dir, "gpu.csv")
sys.stdout = Logger(output_path)

# Create CSV file and write header
csv_path = os.path.join(log_dir, "metrics.csv")
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "dice", "epoch", "samples_read"])

# Declare wandb runtime
if use_wandb: 
    stop_event = threading.Event()
    wandb_run = wandb.init(project=wandb_project, name=timestamp)       
    # Create a separate thread for GPU monitoring
    gpu_monitor_thread = threading.Thread(
        target=gpu_monitor, args=(wandb_run, gpu_csv_path, 0.1, stop_event))
    gpu_monitor_thread.start()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model, loss function, and optimizer
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device).to(dtype)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Create the dataset and dataloader
# dataset = SynthsegDataset(num_samples=num_samples)
# dataset = RandomDataset(num_samples=num_samples) # for debugging 
dataset = MongoTupleheadDataset(config_path = "config.yaml")
dataloader = DataLoader(dataset,
                        batch_size=batch_size, 
                        num_workers=num_generators, pin_memory=True)

samples_read = 0
# Training loop
for epoch in range(num_epochs):
    batch_idxes = [[i] for i in range(num_samples)]
    for batch_idx in batch_idxes:
        inputs, labels = dataset[batch_idx][0]

        inputs = inputs.unsqueeze(0).unsqueeze(0).to(device).to(dtype)  # Add channel dimension
        labels = labels.unsqueeze(0).to(device).to(dtype)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        dice = 1 - loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del inputs
        del labels

        # Update samples read
        samples_read += batch_size
        current_time = time.time()
        # Save metrics to CSV and wandb
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, dice, epoch, samples_read])
        if use_wandb:
            wandb.log({"time": current_time, 
                       "dice": dice, 
                       "epoch": epoch, 
                       "samples_read": samples_read})
        # Print progress
        if (batch_idx[0] + 1) % 10 == 0: 
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save to logdir
torch.save(model.state_dict(), model_path)
shutil.copy("train.py", train_script_path)
# Signal the GPU monitoring thread to stop
stop_event.set()
gpu_monitor_thread.join()
print(f"Model weights, train.py script and output saved in: {log_dir}")
