import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import UNet
from dice import DiceLoss


# Hyperparameters
num_epochs = 10
batch_size = 1
learning_rate = 0.001
n_channels = 1
n_classes = 2

# Custom dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n = 64
        shape = (n,n,n)
        input_data = torch.rand(*shape, dtype=torch.float32)
        label_data = torch.randint(0, 2, shape, dtype=torch.int32)
        return input_data, label_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the dataset and dataloader
dataset = RandomDataset(num_samples=100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1).to(device)  # Add channel dimension
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')
