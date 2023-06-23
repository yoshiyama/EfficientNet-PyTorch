#python transfer1.py --epochs 20 --add_name GAPED --csv_path /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI/Train_Val

#python transfer.py --epochs 20 --add_name OASIS --csv_path /mnt/c/Users/survey/Desktop/OASIS/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/OASIS/Train_Val


#python transfer1.py --epochs 20 --add_name NAPS --csv_path /mnt/c/Users/survey/Desktop/NAPS/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/NAPS/Train_Val

# python transfer1.py --epochs 20 --add_name GAPED_arousal --csv_path /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI/Train_Val

#python transfer.py --epochs 20 --add_name OASIS_arousal --csv_path /mnt/c/Users/survey/Desktop/OASIS/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/OASIS/Train_Val

#python transfer1.py --epochs 20 --add_name NAPS_arousal --csv_path /mnt/c/Users/survey/Desktop/NAPS/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/NAPS/Train_Val

# python transfer1.py --epochs 20 --add_name GAPED --csv_path /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI/Train_Val.csv --traindir /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI/Train_Val --label_type arousal


# 実行方法
import torch
import os
import argparse
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class RegressionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.label_column = 2 if args.label_type == 'arousal' else 1
        # print(self.data.head())
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("idx=",idx)
        img_filename = self.data.iloc[idx,0]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # label = self.data.iloc[idx,1]# 1列目(Valence)の値を取得
        # label = self.data.iloc[idx, 2]  # 2列目(Arousal)の値を取得
        label = self.data.iloc[idx, self.label_column]
        label = float(label)  # Ensure the label is a float
        # print("label=",label)

        return image, label

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run')
parser.add_argument('--add_name', type=str, required=True, help='Additional name for output files')
parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing continuous values')
parser.add_argument('--traindir', type=str, required=True, help='Path to the training directory containing images')
parser.add_argument('--label_type', choices=['valence', 'arousal'], required=True, help='Type of label (valence or arousal)')

args = parser.parse_args()

# add_name="NAPS"
add_name = args.add_name

# Data loading code
# traindir = '/mnt/c/Users/survey/Desktop/NAPS/Train_Val'
traindir = args.traindir

# Load the CSV file containing continuous values
# csv_path = '/mnt/c/Users/survey/Desktop/NAPS/Train_Val.csv'
csv_path = args.csv_path

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_size = EfficientNet.get_image_size('efficientnet-b4')

# continuous_values = pd.read_csv(csv_path)['Valence'].values

# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(image_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))

train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(image_size),
    transforms.Resize((380, 380)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    normalize,
])

train_dataset = RegressionDataset(csv_path, traindir, transform=train_transform)
# print("66line")
# oi,ykk= next(iter(train_dataset))
# print("ykk=",ykk)

# Initialize the KFold class
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

# Initialize the dict to store loss history for each fold
loss_history = {}# Add this line to record losses

# Start the KFold training
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    # Initialize the lists to record losses for this fold
    loss_history[fold] = {"train": [], "val": []}
    train_ids = list(train_ids)
    val_ids = list(val_ids)
    print(f'FOLD {fold}')

    train_subsampler = Subset(train_dataset, train_ids)
    val_subsampler = Subset(train_dataset, val_ids)
    # print('---------------yo-----------------')
    # data_0 = train_subsampler[0]
    # val_0=val_subsampler[0]
    # print('---------------oi-----------------')
    # print("train_subsampler=",len(train_subsampler))

    # Define data loaders for training and validation
    train_loader = DataLoader(train_subsampler, batch_size=16)
    val_loader = DataLoader(val_subsampler, batch_size=16)

    # Initialize the pretrained EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = torch.nn.Linear(model._fc.in_features, 1) # Change for regression task
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss() # Use MSE for regression task
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(type(train_loader))
    # print("line104=",len(train_loader))
    # i,data = next(iter(train_loader))
    # print(data)

    # for i, data in enumerate(train_loader, 0):
    #     print(type(data))
    #     if i > 5:  # 最初の5バッチだけをプリントしてループを終了
    #         break

    # Training process
    # for epoch in range(10):  # loop over the dataset multiple times
    for epoch in range(args.epochs):
        print("epoch=",epoch)
        model.train()
        running_loss = 0.0
        # for i, data in enumerate(train_loader, 0):
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}, Training"):
            # print("i=",i)
            # print("data[1]=",data[1])
            inputs, labels = data[0].to(device), data[1].float().to(device)  # Adjust labels for regression task

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))  # Adjust labels for regression task
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        loss_history[fold]["train"].append(running_loss)  # Record the loss for this fold
        # Validation process
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{args.epochs}, Validation"):
                inputs, labels = data[0].to(device), data[1].float().to(device)  # Adjust labels for regression task

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))  # Adjust labels for regression task
                val_loss += loss.item()

        val_loss /= len(val_loader)
        loss_history[fold]["val"].append(val_loss)  # Record the loss for this fold

        # loss_history["val"].append(val_loss)  # Add this line

        # print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')
        print(f'Fold {fold}, Epoch {epoch + 1}, Training Loss: {running_loss}, Validation Loss: {val_loss}')

    # Save the model after each fold
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = f'efficientnet-b4_fold_{add_name}_{fold}_{current_time}.pth'
    torch.save(model.state_dict(), model_path)
    # torch.save(model.state_dict(), f'efficientnet-b4_fold_{fold}.pth')

print('Finished Training')

current_time_2nd = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# 保存するディレクトリを作成
output_dir = f"results_{add_name}_{current_time_2nd}"
os.makedirs(output_dir, exist_ok=True)

# Plot the loss history for each fold
for fold in loss_history.keys():
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history[fold]["train"], label="Train Loss")
    plt.plot(loss_history[fold]["val"], label="Validation Loss")
    # plt.title(f"Loss history for fold {fold}")
    plt.title(f"Loss history for fold {fold} ({args.label_type} label)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # グラフを画像ファイルとして保存
    plot_filename = f"{output_dir}/loss_history_fold_{add_name}_{fold}_{current_time_2nd}.png"
    plt.savefig(plot_filename)
    plt.show()

# 損失履歴データをCSVファイルとして保存
loss_history_df = pd.DataFrame(loss_history)
# loss_history_filename = f"{output_dir}/loss_history_{add_name}_{current_time_2nd}.csv"
loss_history_filename = f"{output_dir}/loss_history_{args.add_name}_{args.label_type}_{current_time_2nd}.csv"
loss_history_df.to_csv(loss_history_filename, index=False)