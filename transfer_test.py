import torch
import os
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import matplotlib.pyplot as plt


class RegressionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
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

        label = self.data.iloc[idx,1]
        label = float(label)  # Ensure the label is a float
        # print("label=",label)

        return image, label

image_size = EfficientNet.get_image_size('efficientnet-b4')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    # transforms.Resize(image_size),
    transforms.Resize((380, 380)),  #
    transforms.ToTensor(),
    normalize,
])
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the CSV file containing continuous values
test_csv_path = '/mnt/c/Users/survey/Desktop/NAPS/Test.csv'
test_dir='/mnt/c/Users/survey/Desktop/NAPS/Test'

# Assuming you have a separate test dataset and its corresponding data loader
test_dataset = RegressionDataset(test_csv_path, test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16)
model_path = '/mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_0.pth'

# Load the trained model
model = EfficientNet.from_pretrained('efficientnet-b4')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Evaluate the model on the test data
model.eval()
test_loss = 0.0
predictions = []
labels = []
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, target = data[0].to(device), data[1].float().to(device)

        outputs = model(inputs)
        # loss = criterion(outputs, target.unsqueeze(1))
        # test_loss += loss.item()

        predictions.extend(outputs.squeeze().cpu().numpy())
        labels.extend(target.cpu().numpy())

# Calculate evaluation metrics (e.g., RMSE, MAE)
predictions = np.array(predictions)
labels = np.array(labels)

# Print predictions and labels
for i in range(len(predictions)):
    print(f"Sample {i + 1}: Prediction: {predictions[i]}, Label: {labels[i]}")


rmse = np.sqrt(np.mean((predictions - labels) ** 2))
mae = np.mean(np.abs(predictions - labels))

print(f'Test Loss: {test_loss / len(test_loader)}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Plot predictions vs labels
plt.scatter(labels, predictions, color='blue')
plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')
plt.xlabel('Labels')
plt.ylabel('Predictions')
plt.title('Predictions vs Labels')
plt.show()