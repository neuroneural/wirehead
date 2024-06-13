import os
from datetime import datetime
import shutil
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.model import UNet
from utils.dice import DiceLoss
from utils.misc import RandomDataset, Logger
from utils.generator import SynthsegDataset

# Save model, config and output
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_dir = f"./log/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "unet_model.pth")
train_script_path = os.path.join(log_dir, "train.py")
output_path = os.path.join(log_dir, "output.txt")
sys.stdout = Logger(output_path)


# Hyperparameters
num_epochs = 10
batch_size = 1
learning_rate = 0.001
n_channels = 1
n_classes = 2

num_samples = 100
dtype = torch.bfloat16

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model, loss function, and optimizer
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device).to(dtype)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Create the dataset and dataloader
dataset = SynthsegDataset(num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1).to(device).to(dtype)  # Add channel dimension
        labels = labels.to(device).to(dtype)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del inputs
        del labels

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")


# Save to logdir
torch.save(model.state_dict(), model_path)
shutil.copy("train.py", train_script_path)

print(f"Model weights, train.py script and output saved in: {log_dir}")
