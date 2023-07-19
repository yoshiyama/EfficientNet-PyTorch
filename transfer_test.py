# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/NAPS --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_0.pth

# 実行方法
# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_GAPED_3_20230516124343.pth

# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/OASIS --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_OASIS_4_20230514151232.pth




#python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI --model_path efficientnet-b4_fold_GAPED_arousal_0_20230623000612.pth

#python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/OASIS --model_path efficientnet-b4_fold_OASIS_arousal_0_20230623052223.pth


# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/NAPS --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_0.pth --arousal

# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/NAPS --model_path /mnt/c/Users/surveuments/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_0.pth --arousal

# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/NAPS --label_type arousal --output arousal_results_NAPS_0.csv --eval_output arousal_eval_metrics_NAPS_0.csv --plot_output arousal_NAPS_0_predictions_vs_labels.png --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/trained_params_arousal/efficientnet-b4_fold_NAPS_arousal_0_20230623063052.pth


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
import argparse
from scipy.stats import pearsonr


# Add argparse for command line arguments
parser = argparse.ArgumentParser(description='Regression model with EfficientNet')
parser.add_argument('--root_dir', required=True, help='Root directory path')
parser.add_argument('--model_path', required=True, help='Model file path')
parser.add_argument('--label_type', choices=['valence', 'arousal'], default='valence', help='Type of label (valence or arousal)')
parser.add_argument('--output', default='results.csv', help='Output CSV file path')
# Add a new argument for the evaluation metrics output filename
parser.add_argument('--eval_output', default='eval_metrics.csv', help='Evaluation metrics output CSV file path')
# Add an argument for the plot output filename
parser.add_argument('--plot_output', default='predictions_vs_labels.png', help='Plot output file path')

args = parser.parse_args()

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

        label = self.data.iloc[idx, 1 if args.label_type == 'valence' else 2]
        # label = self.data.iloc[idx,1]# 1列目(Valence)の値を取得
        # label = self.data.iloc[idx, 2]  # 2列目(Arousal)の値を取得
        label = float(label)  # Ensure the label is a float
        # print("label=",label)

        return image, label,img_filename

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
# root_dir = '/mnt/c/Users/survey/Desktop/NAPS'
root_dir = args.root_dir
# Load the CSV file containing continuous values
test_csv_path = os.path.join(root_dir, "Test.csv")
test_dir = os.path.join(root_dir, "Test")

print("test_csv_path:", test_csv_path)
print("test_dir:", test_dir)
# test_csv_path = '/mnt/c/Users/survey/Desktop/NAPS/Test.csv'
# test_dir='/mnt/c/Users/survey/Desktop/NAPS/Test'

# Assuming you have a separate test dataset and its corresponding data loader
test_dataset = RegressionDataset(test_csv_path, test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16)
model_path = args.model_path
# model_path = '/mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_0.pth'

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

# Create an empty list to store results
results = []


with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, target, filenames = data[0].to(device), data[1].float().to(device), data[2]
        # inputs, target = data[0].to(device), data[1].float().to(device)

        outputs = model(inputs)
        # loss = criterion(outputs, target.unsqueeze(1))
        # test_loss += loss.item()

        outputs = outputs.squeeze().cpu().numpy()
        target = target.cpu().numpy()

        for j in range(len(outputs)):
            print(f"Sample {filenames[j]}: Prediction: {outputs[j]}, Label: {target[j]}")
            results.append([filenames[j], outputs[j], target[j]])

            # Append the output and target to the predictions and labels lists
            # predictions.append(outputs[j])
            # labels.append(target[j])
            predictions.extend(outputs)
            labels.extend(target)

# Calculate evaluation metrics (e.g., RMSE, MAE)
predictions = np.array(predictions)
labels = np.array(labels)



# Print predictions and labels
# for i in range(len(predictions)):
#     print(f"Sample {i + 1}: Prediction: {predictions[i]}, Label: {labels[i]}")
#     results.append([i + 1, predictions[i], labels[i]])

# After your loop, convert your list to a DataFrame and save it as a CSV file
df = pd.DataFrame(results, columns=['Sample', 'Prediction', 'Label'])

# Add new metrics to the DataFrame
# metrics_df = pd.DataFrame({
#     'Metric': ['Test Loss', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Correlation Coefficient'],
#     'Value': [test_loss / len(test_loader), rmse, mae, correlation]
# })

# Concatenate the original DataFrame and the metrics DataFrame
# df = pd.concat([df, metrics_df], axis=0)

df.to_csv(args.output, index=False)

# Extract your predictions and labels from your DataFrame
predictions = df['Prediction'].to_numpy()
labels = df['Label'].to_numpy()


# Now, you can calculate RMSE, MAE, and the correlation coefficient
rmse = np.sqrt(np.mean((predictions - labels) ** 2))
mae = np.mean(np.abs(predictions - labels))

# Calculate correlation coefficient
correlation, _ = pearsonr(predictions, labels)

# print(f'Test Loss: {test_loss / len(test_loader)}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Correlation Coefficient: {correlation}')

# Plot predictions vs labels
plt.scatter(labels, predictions, color='blue')
plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')
plt.xlabel('Labels')
plt.ylabel('Predictions')
plt.title('Predictions vs Labels')
# plt.show()
# plt.savefig('predictions_vs_labels.png')
# Save the figure
plt.savefig(args.plot_output)

# Create another DataFrame to store evaluation metrics
eval_metrics = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'Correlation Coefficient'],
    'Value': [rmse, mae, correlation]
})

# Save evaluation metrics as a CSV file
# eval_metrics.to_csv('eval_metrics.csv', index=False)
eval_metrics.to_csv(args.eval_output, index=False)