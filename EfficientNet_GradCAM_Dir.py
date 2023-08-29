# python EfficientNet_GradCAM_Dir.py --model_path path/to/your/model.pth --image_dir path/to/your/image/directory


import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Grad-CAM on EfficientNet')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--image_dir', type=str, required=True, help='Directory path containing the images')
args = parser.parse_args()

# Initialize the EfficientNet model
model = EfficientNet.from_name('efficientnet-b4')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Results list for storing inference values
results = []

for image_name in os.listdir(args.image_dir):
    image_path = os.path.join(args.image_dir, image_name)

    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(380),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    target_layer = model._conv_head
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_batch)
    output_value = output.item()  # Store the inference value
    results.append((image_name, output_value))
    output.backward()

    # ... [Same Grad-CAM code as above] ...
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
    # Normalize and resize
    positive_cam = cv2.resize(positive_cam, (input_image.size[0], input_image.size[1]))
    #change below
    positive_cam = (positive_cam - positive_cam.min()) / (positive_cam.max() - positive_cam.min())

    ## After:
    # max_value = np.percentile(positive_cam, 90)
    # positive_cam = np.clip(positive_cam, 0, max_value) / max_value
    ####

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

    # plt.show()

    # Save the overlayed images
    image_basename, _ = os.path.splitext(image_name)  # Split the name and extension
    pos_save_path = os.path.join(args.image_dir, f"{image_basename}_pos_gradcam.jpg")
    neg_save_path = os.path.join(args.image_dir, f"{image_basename}_neg_gradcam.jpg")
    cv2.imwrite(pos_save_path, cv2.cvtColor(overlayed_pos_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(neg_save_path, cv2.cvtColor(overlayed_neg_image, cv2.COLOR_RGB2BGR))
    # pos_save_path = os.path.join(args.image_dir, f"{image_name}_pos_gradcam.jpg")
    # neg_save_path = os.path.join(args.image_dir, f"{image_name}_neg_gradcam.jpg")
    # cv2.imwrite(pos_save_path, cv2.cvtColor(overlayed_pos_image, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(neg_save_path, cv2.cvtColor(overlayed_neg_image, cv2.COLOR_RGB2BGR))

# Save the inference values
directory_name = os.path.basename(args.image_dir)
results_path = os.path.join(args.image_dir, f'{directory_name}.csv')
with open(results_path, 'w') as f:
    f.write("ImageName,InferenceValue\n")
    for image_name, value in results:
        f.write(f"{image_name},{value}\n")
