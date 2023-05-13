import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading code
traindir = 'path_to_your_data'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_size = EfficientNet.get_image_size('efficientnet-b4')

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

# Initialize the KFold class
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start the KFold training
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Define the data subsets for training and validation
    train_subsampler = Subset(train_dataset, train_ids)
    val_subsampler = Subset(train_dataset, val_ids)

    # Define data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size=16, sampler=val_subsampler)

    # Initialize the pretrained EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = torch.nn.Linear(model._fc.in_features, 1) # Change for regression task
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss() # Use MSE for regression task
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training process
    for epoch in range(10):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].float().to(device)  # Adjust labels for regression task

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))  # Adjust labels for regression task
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation process
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].float().to(device)  # Adjust labels for regression task

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))  # Adjust labels for regression task
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')

    # Save the model after each fold
    torch.save(model.state_dict(), f'efficientnet-b4_fold_{fold}.pth')

print('Finished Training')