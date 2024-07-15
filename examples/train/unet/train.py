import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.model import UNet
from utils.dice import DiceLoss
from wirehead import MongoTupleheadDataset

WIREHEAD_CONFIG = "./config.yaml"

# Hyperparameters
batch_size = 1    
learning_rate = 1e-4
n_channels = 1
n_classes = 18
num_samples = 1 
num_epochs = 1 
dtype = torch.float16
# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Initialize the model, loss function, and optimizer
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device).to(dtype)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dataset = MongoTupleheadDataset(config_path = WIREHEAD_CONFIG)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        pin_memory=True)

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

        if (batch_idx[0] + 1) % 1 == 0: 
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx[0]+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
