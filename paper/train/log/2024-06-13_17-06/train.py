import os
from datetime import datetime
import shutil
import sys
import csv
import time
import subprocess


import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.model import UNet
from utils.dice import DiceLoss
from utils.misc import Logger, RandomDataset
from utils.generator import SynthsegDataset


# Save model, config and output
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Logging
log_dir = f"./log/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "unet_model.pth")
train_script_path = os.path.join(log_dir, "train.py")
output_path = os.path.join(log_dir, "output.txt")
sys.stdout = Logger(output_path)
# Create CSV file and write header
csv_path = os.path.join(log_dir, "metrics.csv")
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "gpu_util", "gpu_mem", "dice", "epoch", "samples_read"])

use_wandb = True 
if use_wandb: 
    wandb.init(project="wirehead_1x3090_"+timestamp)       

# Hyperparameters
num_epochs = 1
batch_size = 1
learning_rate = 0.001
n_channels = 1
n_classes = 2

num_samples = 100       # number of samples to generate per epoch
dtype = torch.bfloat16  

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model, loss function, and optimizer
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device).to(dtype)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Create the dataset and dataloader
# dataset = SynthsegDataset(num_samples=num_samples)
dataset = RandomDataset(num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
samples_read = 0
start_time = time.time()

for epoch in range(num_epochs):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1).to(device).to(dtype)  # Add channel dimension
        labels = labels.to(device).to(dtype)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        dice = 1 - loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del inputs
        del labels

        # Get current time and GPU utilization and memory usage
        gpu_info = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=utilization.gpu,memory.used", 
            "--format=csv,nounits,noheader"])
        gpu_info = gpu_info.decode('utf-8').strip().split(',')
        gpu_util = float(gpu_info[0])
        gpu_mem = float(gpu_info[1])
        # Update samples read
        samples_read += batch_size

        current_time = time.time() - start_time
        # Save metrics to CSV and wandb
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, gpu_util, gpu_mem, dice, epoch, samples_read])
        if use_wandb:
            wandb.log({"time": current_time, 
                       "gpu_util": gpu_util, 
                       "gpu_mem": gpu_mem, 
                       "dice": dice, 
                       "epoch": epoch, 
                       "samples_read": samples_read})
        # Print progress
        if (batch_idx + 1) % 10 == 0: 
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save to logdir
torch.save(model.state_dict(), model_path)
shutil.copy("train.py", train_script_path)
print(f"Model weights, train.py script and output saved in: {log_dir}")
