import os 
from datetime import datetime
import shutil
import sys
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.model import UNet
from utils.dice import faster_dice, DiceLoss
from utils.logging import Logger
from utils.fetch import get_eval
from wirehead import MongoTupleheadDataset
from wirehead.dataset import unit_interval_normalize

### Userland ###
WIREHEAD_CONFIG = "./config.yaml"

# Hyperparameters
batch_size = 1         # this should be 1 to match synthseg
learning_rate = 1e-4   # this should be 1 to match synthseg
n_channels = 1         # unclear
n_classes = 18         # unclear 
num_samples = 10
num_epochs = 1000      # 100*10 = 1000
num_generators = 1     # unclear
dtype = torch.float32
### outside ###

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Logging constants
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = f"./log/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    model_path          = os.path.join(log_dir, f"unet_model_rank{rank}.pth")
    train_script_path   = os.path.join(log_dir, "train_ddp.py")
    output_path         = os.path.join(log_dir, f"output_rank{rank}.txt")
    sys.stdout = Logger(output_path)

    # Create CSV file and write header
    csv_path = os.path.join(log_dir, f"metrics_rank{rank}.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time", "dice", "epoch", "samples_read"])

    # Device configuration
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model, loss function, and optimizer
    model = UNet(n_channels=n_channels, n_classes=n_classes).to(device).to(dtype)
    model = DDP(model, device_ids=[rank])
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create the dataset and dataloader
    dataset = MongoTupleheadDataset(config_path = WIREHEAD_CONFIG)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_generators, 
                            pin_memory=True)

    # Get some real brains from HCPnew to eval
    eval_set = get_eval(10)
    print(f"Training: Got {len(eval_set)} samples for testing")

    samples_read = 0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.unsqueeze(1).to(device).to(dtype)  # Add channel dimension
            labels = labels.to(device).to(dtype)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            dice = 1 - loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update samples read
            samples_read += batch_size * world_size
            current_time = time.time()
            # Save metrics to CSV
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_time, dice, epoch, samples_read])
            
            # Print progress
            if (batch_idx + 1) % 1 == 0 and rank == 0: 
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Test on real data every epoch
        model.eval()
        with torch.inference_mode():
            eval_dices = []
            for img, lab in eval_set:
                img = img.to(device)
                out = model(img)
                out = torch.squeeze(torch.argmax(out, 1)).long()
                lab = torch.squeeze(lab)
                eval_dice = torch.mean(
                    faster_dice(out, lab, range(n_classes))
                )
                eval_dices.append(eval_dice)

            if rank == 0:
                print(f"Eval: Average dice: {sum(eval_dices)/len(eval_dices)}")

    # Save to logdir
    if rank == 0:
        torch.save(model.state_dict(), model_path)
        shutil.copy("train_ddp.py", train_script_path)
        print(f"Model weights, train_ddp.py script and output saved in: {log_dir}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
