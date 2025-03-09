#!/usr/bin/env python
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from model_core.models import SegmentationModel
from model_core.utils import decode_segmap, create_binary_mask, create_ground_truth_binary_mask

# Define color settings (should match those used during training)
colors = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [250, 170, 30],
    [220, 220, 0],
    [220, 20, 60],
    [0, 0, 142],
    [119, 11, 32],
]
num_classes = 8
label_colors = dict(zip(range(num_classes), colors))

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python inference.py <path/to/model> <path/to/images> <path/to/labels> "
            "<output_folder_segmentation> <output_folder_binary> <output_folder_ground_truth>"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    images_folder = sys.argv[2]
    labels_folder = sys.argv[3]
    output_seg_folder = sys.argv[4]
    output_binary_folder = sys.argv[5]
    output_ground_truth_folder = sys.argv[6]
    rgb = [244, 35, 232]

    os.makedirs(output_seg_folder, exist_ok=True)
    os.makedirs(output_binary_folder, exist_ok=True)
    os.makedirs(output_ground_truth_folder, exist_ok=True)

    # Define transformation for input images
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.432, 0.433, 0.424), (0.263, 0.264, 0.278))
    ])

    # Generate ground truth binary masks
    create_ground_truth_binary_mask(labels_folder, output_ground_truth_folder, rgb, target_size=(512, 256))

    # Load the segmentation model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    test_images = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]

    with torch.no_grad():
        print("****** Creating Segmentation and Binary Masks ******")
        for image_file in tqdm(test_images):
            image_path = os.path.join(images_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            output_cpu = output.detach().cpu()[0]
            pred = torch.argmax(output_cpu, 0)
            decoded_output = decode_segmap(pred.numpy(), label_colors)
            # Convert the decoded output to a 0-255 image format
            decoded_output = (decoded_output * 255).clip(0, 255).astype(np.uint8)

            binary_mask = create_binary_mask(decoded_output, rgb)

            # Save the segmentation mask and binary mask
            seg_output_path = os.path.join(output_seg_folder, f"seg_mask_{image_file}")
            binary_output_path = os.path.join(output_binary_folder, image_file.replace('.jpg', '.png'))
            Image.fromarray(decoded_output).save(seg_output_path)
            Image.fromarray(binary_mask).save(binary_output_path)

    print("Segmentation masks saved in:", output_seg_folder)
    print("Binary masks saved in:", output_binary_folder)
