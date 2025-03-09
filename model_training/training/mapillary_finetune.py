#!/usr/bin/env python

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.utils.data import DataLoader

from model_core.data import MapillaryDataset, get_transform
from model_core.models import SegmentationModel
from model_core.train_utils import train_fn, eval_fn
from model_core.utils import plot_losses, encode_segmap

# ---------------------------
# Configuration parameters for Mapillary fine-tuning
# ---------------------------
MAPILLARY_ROOT = "Mapillary-Vistas-1000-sidewalks"
MAPILLARY_MASKS = "Mapillary_converted_masks"
EPOCHS = 100
LEARNING_RATE = 0.0001
ENCODER = "resnet34"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMG_SIZE = (256, 512)
CHECKPOINT_FOLDER = "mapillary_checkpoints"
LOG_FILE_PATH = "mapillary_training_log.txt"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, "mapillary_checkpoint.pt")

# ---------------------------
# Dataset specifics for Mapillary
# ---------------------------
# These values come from your original fine-tuning script.
VOID_CLASSES = [1,2,3,5,6,7,4,8,9,10,11,12,13,15,17,18,19,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,52,54,57,60,61,63,64,65,28,21,62,55,59,58]
TARGET_CLASSES = [66, 14, 16, 49, 51, 53, 56, 20]
NUM_CLASSES = len(TARGET_CLASSES)
LABEL_COLORS = {
    0: [0, 0, 0],
    1: [128, 64, 128],
    2: [244, 35, 232],
    3: [250, 170, 30],
    4: [220, 220, 0],
    5: [220, 20, 60],
    6: [0, 0, 142],
    7: [119, 11, 32],
}

# For Mapillary, void classes are set to 66 (as in your original script)
encode_fn = lambda mask: encode_segmap(mask, VOID_CLASSES, TARGET_CLASSES, 66)

# ---------------------------
# Define transformation
# ---------------------------
transform = get_transform(IMG_SIZE)

# ---------------------------
# Prepare Datasets and DataLoaders
# ---------------------------
train_images = os.path.join(MAPILLARY_ROOT, "training", "images")
val_images = os.path.join(MAPILLARY_ROOT, "validation", "images")
train_masks = os.path.join(MAPILLARY_MASKS, "training")
val_masks = os.path.join(MAPILLARY_MASKS, "validation")

train_dataset = MapillaryDataset(image_folder=train_images, mask_folder=train_masks, transform=transform)
val_dataset = MapillaryDataset(image_folder=val_images, mask_folder=val_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Load pre-trained model and adjust segmentation head if necessary
# ---------------------------
model = SegmentationModel(encoder=ENCODER, num_classes=NUM_CLASSES)
# Load base model weights (trained on Cityscapes)
model.load_state_dict(torch.load("base_model.pt", map_location=DEVICE))
# Adjust segmentation head (if the first layer needs to match NUM_CLASSES)
model.layer.segmentation_head[0] = nn.Conv2d(
    model.layer.segmentation_head[0].in_channels, NUM_CLASSES, kernel_size=1
)
model.to(DEVICE)

criterion = smp.losses.DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)

# ---------------------------
# Load Checkpoint if available
# ---------------------------
start_epoch = 1
best_valid_loss = np.Inf
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_valid_loss = checkpoint["best_valid_loss"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed training from epoch {start_epoch}")

train_losses = []
valid_losses = []

with open(LOG_FILE_PATH, "w") as log_file:
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"Epoch: {epoch}")

        train_loss = train_fn(train_loader, model, optimizer, criterion, DEVICE, encode_fn)
        valid_loss = eval_fn(val_loader, model, criterion, DEVICE, encode_fn)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_valid_loss": valid_loss,
            }
            torch.save(checkpoint_dict, CHECKPOINT_PATH)
            torch.save(model.state_dict(), "fine_tuned_model.pt")
            print("Model Saved!")
            best_valid_loss = valid_loss

        log_file.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}\n")
        print("\n")

print(f"Training log saved to {LOG_FILE_PATH}")
plot_losses(train_losses, valid_losses, "Training Loss on Mapillary Dataset", "finetuned_loss_fig.png")
