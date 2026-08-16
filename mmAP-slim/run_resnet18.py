import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

import wandb
USE_WANDB = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data directories
train_dir = 'finetune-norm128-resnet/train'
eval_dir = 'finetune-norm128-resnet/eval'

# Define transformations for the training and evaluation datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
eval_dataset = datasets.ImageFolder(root=eval_dir, transform=transform)

# Hyperparameters and configuration variables
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MOMENTUM = 0.9

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model
model = models.resnet18(weights=None)  # No pretrained weights
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer to match the number of classes

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Initialize a new W&B run if USE_WANDB is True and wandb is available
if USE_WANDB and wandb:
    wandb.init(project='resnet18_angle-doppler-range', name='resnet18_noaug-rcd-norm_20e_lr001_angle-doppler-range_cls')
    wandb.config.update({
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
    })

# Training function with tqdm for progress bar
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Use tqdm to wrap the train_loader for progress bar display
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        if USE_WANDB and wandb:
            wandb.log({"epoch": epoch + 1, "loss": epoch_loss})

        evaluate_model(model, eval_loader)

# Evaluation function with tqdm for progress bar
def evaluate_model(model, eval_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Use tqdm to wrap the eval_loader for progress bar display during evaluation
        for inputs, labels in tqdm(eval_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    if USE_WANDB and wandb:
        wandb.log({"accuracy": accuracy})

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

if USE_WANDB and wandb:
    wandb.finish()
