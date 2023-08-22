# python EfficientNet_GradCAM.py --model_path path/to/your/trained/model.pth --image_path path/to/your/image.jpg

# python EfficientNet_GradCAM.py --model_path /mnt/c/Users/survey/Documents/GitHub/EfficientNet-PyTorch/trained_params_valence/efficientnet-b4_fold_OASIS_valence_1.pth --image_path /mnt/c/Users/survey/Desktop/景観_ 電柱/2022/Toyota_pole_eval_images-extracted/no_pole_2048/output_108.jpg

import argparse
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Or 'TkAgg', 'WebAgg', etc.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Grad-CAM on EfficientNet')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
args = parser.parse_args()

# Initialize the EfficientNet model
model = EfficientNet.from_name('efficientnet-b4')
model._fc = torch.nn.Linear(model._fc.in_features, 1)  # Change for regression task

# Load the trained weights
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Load and preprocess the input image
input_image = Image.open(args.image_path)
preprocess = transforms.Compose([
    transforms.Resize(380),
    transforms.ToTensor(),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Register hooks for Grad-CAM
activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations['value'] = output

def backward_hook(module, grad_input, grad_output):
    gradients['value'] = grad_output[0]

target_layer = model._conv_head  # Using the last convolutional layer as target
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward and backward pass
model.zero_grad()
output = model(input_batch)
output.backward()  # In regression, backpropagate on the output directly

# Compute Grad-CAM
activation = activations['value'].detach().cpu().numpy()[0]
gradient = gradients['value'].detach().cpu().numpy()[0]

# Compute Grad-CAM for positive contributions
weights = gradient.mean(axis=(1, 2))  # Global Average Pooling
positive_cam = np.sum((activation * weights[:, None, None]), axis=0)
positive_cam = np.maximum(positive_cam, 0)  # ReLU

# Compute Grad-CAM for negative contributions
negative_weights = (gradient * (gradient < 0)).mean(axis=(1, 2))  # Average only negative gradients
negative_cam = np.sum((activation * negative_weights[:, None, None]), axis=0)
negative_cam = np.maximum(negative_cam, 0)  # ReLU

# Normalize and resize
positive_cam = cv2.resize(positive_cam, (input_image.size[0], input_image.size[1]))
positive_cam = (positive_cam - positive_cam.min()) / (positive_cam.max() - positive_cam.min())

negative_cam = cv2.resize(negative_cam, (input_image.size[0], input_image.size[1]))
negative_cam = (negative_cam - negative_cam.min()) / (negative_cam.max() - negative_cam.min())

# 1. Apply the 'jet' colormap on the Grad-CAM heatmap
colored_pos_cam = cm.jet(positive_cam)
colored_neg_cam = cm.jet(negative_cam)

# 2. Convert from 0-1 to 0-255 scale
colored_pos_cam = (colored_pos_cam * 255).astype(np.uint8)
colored_neg_cam = (colored_neg_cam * 255).astype(np.uint8)

# 3. Convert to 3 channels (RGB)
colored_pos_cam = cv2.cvtColor(colored_pos_cam, cv2.COLOR_RGBA2RGB)
colored_neg_cam = cv2.cvtColor(colored_neg_cam, cv2.COLOR_RGBA2RGB)

input_image_np = np.array(input_image)

# 4. Overlay the colored heatmaps on the original image
overlayed_pos_image = cv2.addWeighted(input_image_np, 0.5, colored_pos_cam, 0.5, 0)
overlayed_neg_image = cv2.addWeighted(input_image_np, 0.5, colored_neg_cam, 0.5, 0)

# Display images
plt.figure(figsize=(20, 10))

plt.subplot(1, 5, 1)
plt.title('Input Image')
plt.imshow(input_image_np)
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title('Positive Grad-CAM')
plt.imshow(positive_cam, cmap='jet')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title('Negative Grad-CAM')
plt.imshow(negative_cam, cmap='jet')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title('Overlay Positive')
plt.imshow(overlayed_pos_image)
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title('Overlay Negative')
plt.imshow(overlayed_neg_image)
plt.axis('off')

plt.show()


# # Clip the heatmap values
# threshold_pos = np.percentile(positive_cam, 90)
# positive_cam = np.minimum(positive_cam, threshold_pos)
#
# threshold_neg = np.percentile(negative_cam, 90)
# negative_cam = np.minimum(negative_cam, threshold_neg)
#
# # Reshape to have 3 channels
# positive_cam_3_channels = np.expand_dims(positive_cam, axis=2)
# positive_cam_3_channels = np.tile(positive_cam_3_channels, (1, 1, 3))
#
# negative_cam_3_channels = np.expand_dims(negative_cam, axis=2)
# negative_cam_3_channels = np.tile(negative_cam_3_channels, (1, 1, 3))
#
# input_image_np = np.array(input_image)
#
# # Display images
# plt.figure(figsize=(20, 10))
#
# plt.subplot(141)
# plt.title('Input Image')
# plt.imshow(input_image_np)
# plt.axis('off')
#
# plt.subplot(142)
# plt.title('Positive Grad-CAM')
# plt.imshow(positive_cam_3_channels, cmap='jet')
# plt.axis('off')
#
# plt.subplot(143)
# plt.title('Negative Grad-CAM')
# plt.imshow(negative_cam_3_channels, cmap='jet')
# plt.axis('off')
#
# plt.subplot(144)
# plt.title('Overlay')
# plt.imshow(overlayed_image)
# plt.axis('off')
#
# plt.show()
#
#
# # weights = gradient.mean(axis=(1, 2))  # Global Average Pooling
# # cam = np.sum((activation * weights[:, None, None]), axis=0)
# # cam = np.maximum(cam, 0)  # ReLU
# # cam = cv2.resize(cam, (input_image.size[0], input_image.size[1]))
# # # Clip the heatmap values
# # threshold = np.percentile(cam, 90)  # Clip values above the 99th percentile
# # cam = np.minimum(cam, threshold)
# # cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
# #
# # # Reshape cam to have 3 channels
# # cam_3_channels = np.expand_dims(cam, axis=2)
# # cam_3_channels = np.tile(cam_3_channels, (1, 1, 3))
# #
# # # Overlay Grad-CAM with image
# # input_image_np = np.array(input_image)
# # # print("Shape of input_image_np:", input_image_np.shape)
# # # print("Shape of cam:", cam.shape)
# # # overlayed_image = cv2.addWeighted(input_image_np, 0.5, (cam_3_channels * 255).astype(np.uint8), 0.5, 0)
# #
# # # 1. Apply the 'jet' colormap on the Grad-CAM heatmap
# # colored_cam = cm.jet(cam)
# #
# # # 2. Convert colored_cam from 0-1 to 0-255 scale
# # colored_cam = (colored_cam * 255).astype(np.uint8)
# #
# # # 3. Convert to 3 channels (RGB)
# # colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_RGBA2RGB)
# #
# # # 4. Overlay the colored heatmap on the original image
# # overlayed_image = cv2.addWeighted(input_image_np, 0.5, colored_cam, 0.5, 0)
# #
# # # Display images
# # plt.figure(figsize=(10, 10))
# #
# # plt.subplot(131)
# # plt.title('Input Image')
# # plt.imshow(input_image_np)
# # plt.axis('off')
# #
# # plt.subplot(132)
# # plt.title('Grad-CAM')
# # plt.imshow(cam, cmap='jet')
# # plt.axis('off')
# #
# # plt.subplot(133)
# # plt.title('Overlay')
# # plt.imshow(overlayed_image)
# # plt.axis('off')
# #
# # # plt.show()
# # # plt.savefig('gradcam_results.png')  # Save the figure
# # plt.show()