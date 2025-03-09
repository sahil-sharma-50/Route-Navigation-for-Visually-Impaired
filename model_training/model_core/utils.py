import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def encode_segmap(mask, void_classes, target_classes, ignore_index):
    # Replace void classes with ignore_index
    for void_class in void_classes:
        mask[mask == void_class] = ignore_index
    # Map valid classes to a contiguous range [0, num_classes-1]
    class_label_mapping = {c: i for i, c in enumerate(target_classes)}
    for valid_class in target_classes:
        mask[mask == valid_class] = class_label_mapping[valid_class]
    return mask


def decode_segmap(segmentation_map, label_colors):
    """
    Convert segmentation map to an RGB image using label colors.
    """
    rgb = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8
    )
    for label, color in label_colors.items():
        rgb[segmentation_map == label] = color
    return rgb / 255.0


def create_binary_mask(img, rgb_value):
    """
    Create a binary mask where pixels matching the provided rgb_value are set to 255.
    """
    mask = (
        np.all(img == np.array(rgb_value).reshape(1, 1, 3), axis=2).astype(np.uint8)
        * 255
    )
    return mask


def create_ground_truth_binary_mask(
    labels_dir, output_dir, rgb_value, target_size=(512, 256)
):
    """
    For each image in labels_dir, create a binary mask based on rgb_value and save it to output_dir.
    """
    print("\n****** Creating Ground Truth Binary Masks: ******")
    for filename in tqdm(os.listdir(labels_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(labels_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            binary_mask = create_binary_mask(image, tuple(rgb_value))
            binary_mask = cv2.resize(
                binary_mask, target_size, interpolation=cv2.INTER_NEAREST
            )
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, binary_mask)
    print("Ground truth binary masks saved in:", output_dir, "\n")


def plot_losses(train_losses, valid_losses, title, filename):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(filename)
