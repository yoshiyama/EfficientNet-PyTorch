# 実行方法
# python transfer_test.py --root_dir /mnt/c/Users/survey/Desktop/GAPED_2/GAPED/GAPED4AI --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/efficientnet-b4_fold_GAPED_1_20230514144151.pth
# python transfer_eval.py --image_dir /home/user/images --model_path /home/user/models/model.pth
# python transfer_eval.py --image_dir /mnt/c/Users/keikan2/Desktop/Test/ --model_path /mnt/c/Users/keikan2/Desktop/EfficientNet-PyTorch/trained_params_valence/efficientnet-b4_fold_OASIS_valence_0.pth --output_csv /mnt/c/Users/keikan2/Desktop/test.csv



import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import argparse
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Add argparse for command line arguments
parser = argparse.ArgumentParser(description='Inference with EfficientNet')
parser.add_argument('--image_dir', required=True, help='Image directory path')
parser.add_argument('--model_path', required=True, help='Model file path')
parser.add_argument('--output_csv', required=True, help='Output CSV file name')  # Add this line
args = parser.parse_args()


class InferenceDataset(Dataset):
    # def __init__(self, csv_file, image_dir, transform=None):
    def __init__(self, image_dir, transform=None):
        # self.data = pd.read_csv(csv_file)
        # self.image_dir = image_dir
        # self.transform = transform
        self.image_files = glob.glob(os.path.join(image_dir, '*'))
        # print(os.path.join(image_dir, '*'))
        self.transform = transform
        # print(self.image_files)

    def __len__(self):
        # return len(self.data)
        return len(self.image_files)

    def __getitem__(self, idx):
        # img_filename = self.data.iloc[idx,0]
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

            # Add a batch dimension
            # image = image.unsqueeze(0)

        return image,img_path

image_size = EfficientNet.get_image_size('efficientnet-b4')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    normalize,
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = args.image_dir


# テストデータセットとその対応するデータローダーを想定
test_dataset = InferenceDataset(image_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16)

model_path = args.model_path

# Load the trained model
model = EfficientNet.from_pretrained('efficientnet-b4')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Perform inference
model.eval()
predictions = []
# image_paths = []
image_filenames = []
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, paths = data[0].to(device), data[1]
        outputs = model(inputs)
        predictions.extend(np.atleast_1d(outputs.squeeze().cpu().numpy()))
        image_filenames.extend([os.path.basename(path) for path in paths])
        # image_paths.extend(paths)  # 画像のパスを追加

# # 予測の出力とCSVへの保存
# results = pd.DataFrame({
#     'Image_Path': image_paths,
#     'Prediction': predictions
# })
# 予測の出力とCSVへの保存
results = pd.DataFrame({
    'Image_Filename': image_filenames,  # Change column name to 'Image_Filename'
    'Prediction': predictions
})

# Save to the specified CSV file
results.to_csv(args.output_csv, index=False)  # Change this line

for i in range(len(predictions)):
    print(f"画像ファイル名: {image_filenames[i]}, 予測値: {predictions[i]}")  # Change '画像パス' to '画像ファイル名'

# for i in range(len(predictions)):
#     print(f"画像パス: {image_paths[i]}, 予測値: {predictions[i]}")

# # Generate unique filename
# current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# filename = f'predictions_{current_time}.csv'
#
# results.to_csv(filename, index=False)
#
# for i in range(len(predictions)):
#     print(f"画像パス: {image_paths[i]}, 予測値: {predictions[i]}")