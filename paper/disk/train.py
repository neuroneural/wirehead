import os 
from datetime import datetime
import shutil
import sys
import csv
import time
import argparse
import threading

import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np

from utils.model import UNet
from utils.dice import faster_dice, DiceLoss
from utils.logging import Logger, gpu_monitor 
from utils.fetch import get_eval
from wirehead.dataset import unit_interval_normalize

### Userland ###
use_wandb = True 
wandb_project = "wirehead_1xA100_disk"
WIREHEAD_CONFIG = "./config.yaml"

# Hyperparameters
batch_size = 1         # this should be 1 to match synthseg
learning_rate = 1e-4   # this should be 1e-4 to match synthseg
n_channels = 1         # unclear
n_classes = 18         # unclear 
num_samples = 1000
num_epochs = 1         # 100*10 = 1000
num_generators = 1     # unclear
dtype = torch.float32
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
    parser = argparse.ArgumentParser(description='Run training script with name.')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment (optional)')
    args = parser.parse_args()

    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = timestamp
    print(f"Experiment name: {experiment_name}")
    stop_event = threading.Event()
    wandb_run = wandb.init(project=wandb_project, name=experiment_name)       
    # Create a separate thread for GPU monitoring
    gpu_monitor_thread = threading.Thread(
        target=gpu_monitor, args=(wandb_run, gpu_csv_path, 0.1, stop_event))
    gpu_monitor_thread.start()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# New FakeSampleDataset
class FakeSampleDataset(Dataset):
    def __init__(self, img_path, lab_path, num_samples):
        self.img_path = img_path
        self.lab_path = lab_path
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = nib.load(self.img_path).get_fdata()
        lab = nib.load(self.lab_path).get_fdata()
        img = (img - img.min()) / (img.max() - img.min()) * 255  # normalize to 0-255
        return torch.from_numpy(img).float(), torch.from_numpy(lab).long()

# Initialize the model, loss function, and optimizer
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device).to(dtype)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the dataset and dataloader
PATH_TO_DATA = "./fake_samples/"
dataset = FakeSampleDataset(
    os.path.join(PATH_TO_DATA, "brain_img.nii.gz"),
    os.path.join(PATH_TO_DATA, "brain_lab.nii.gz"),
    num_samples=num_samples
)
dataloader = DataLoader(dataset,
                        batch_size=batch_size, 
                        num_workers=num_generators, pin_memory=True)

# Get some real brains from HCPnew to eval
eval_set = get_eval(10)

print(f"Training: Got {len(eval_set)} samples for testing")

samples_read = 0
# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.unsqueeze(0).to(device).to(dtype)  # Add channel dimension
        inputs = unit_interval_normalize(inputs)
        labels = labels.to(device).to(dtype)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        dice = 1 - loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update samples read
        samples_read += batch_size
        current_time = time.time()
        # Save metrics to CSV and wandb
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, dice, epoch, samples_read])
        if use_wandb:
            result = torch.squeeze(torch.argmax(outputs, 1)).long()
            labels = torch.squeeze(labels)
            real_dice = torch.mean(
                faster_dice(result, labels, range(n_classes))
            ) # use real dice instead of dice loss
            wandb.log({"time": current_time, 
                       "dice": real_dice, 
                       "epoch": epoch, 
                       "samples_read": samples_read})
            
            if (batch_idx + 1) % 10 == 0: # test on eval  
                with torch.inference_mode():
                    eval_dices = []
                    for img, lab in eval_set:
                        img = img.cuda()
                        img = unit_interval_normalize(img) # this is slow and should be preprocessed in the first place
                        out = model(img)
                        out = torch.squeeze(torch.argmax(out, 1)).long()
                        lab = torch.squeeze(lab)
                        eval_dice = torch.mean(
                            faster_dice(out, lab, range(n_classes))
                        )
                        eval_dices.append(eval_dice)

                    wandb.log({"eval_dice": sum(eval_dices)/len(eval_dices)})
                    print(f"Eval: Average dice: {sum(eval_dices)/len(eval_dices)}")

        # Print progress
        if (batch_idx + 1) % 10 == 0: 
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save to logdir
torch.save(model.state_dict(), model_path)
shutil.copy("train.py", train_script_path)
# Signal the GPU monitoring thread to stop
if use_wandb:
    stop_event.set()
    gpu_monitor_thread.join()
print(f"Model weights, train.py script and output saved in: {log_dir}")
