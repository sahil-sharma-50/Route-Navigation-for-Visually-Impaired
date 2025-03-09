#!/usr/bin/env python

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.utils.data import DataLoader
import shutil

from model_core.data import CityscapesDataset, get_transform
from model_core.models import SegmentationModel
from model_core.train_utils import train_fn, eval_fn
from model_core.utils import plot_losses, encode_segmap

# ---------------------------
# Configuration parameters
# ---------------------------
CITYSCAPES_ROOT = "trainvaltest"  # dataset root directory
EPOCHS = 100
LEARNING_RATE = 0.001
ENCODER = "resnet34"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
IMG_SIZE = (256, 512)
CHECKPOINT_FOLDER = "cityscapes_checkpoints"
LOG_FILE_PATH = "cityscapes_training_log.txt"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, "cityscapes_checkpoint.pt")

# ---------------------------
# Data Preparation (if needed)
# ---------------------------
# The following moves folders if they exist (as in your original script)
gtFine_source_folder = os.path.join(CITYSCAPES_ROOT, "gtFine_trainvaltest", "gtFine")
leftImg8bit_source_folder = os.path.join(CITYSCAPES_ROOT, "leftImg8bit_trainvaltest", "leftImg8bit")
gtFine_target_folder = os.path.join(CITYSCAPES_ROOT, "gtFine")
leftImg8bit_target_folder = os.path.join(CITYSCAPES_ROOT, "leftImg8bit")
try:
    shutil.move(gtFine_source_folder, gtFine_target_folder)
    shutil.move(leftImg8bit_source_folder, leftImg8bit_target_folder)
except Exception:
    pass

# ---------------------------
# Dataset specifics
# ---------------------------
IGNORE_INDEX = 255
VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1, 11, 12, 13, 17, 21, 22, 23, 25, 27, 28, 31, 32]
TARGET_CLASSES = [IGNORE_INDEX, 7, 8, 19, 20, 24, 26, 33]
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

# Use a lambda to pass the proper parameters to the encoding function
encode_fn = lambda mask: encode_segmap(mask, VOID_CLASSES, TARGET_CLASSES, IGNORE_INDEX)

# Define transformation
transform = get_transform(IMG_SIZE)

# ---------------------------
# Prepare Datasets and DataLoaders
# ---------------------------
trainset = CityscapesDataset(
    root=CITYSCAPES_ROOT, split="train", mode="fine", target_type="semantic", transform=transform
)
valset = CityscapesDataset(
    root=CITYSCAPES_ROOT, split="val", mode="fine", target_type="semantic", transform=transform
)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model, Loss, and Optimizer
# ---------------------------
model = SegmentationModel(encoder=ENCODER, num_classes=NUM_CLASSES)
model.to(DEVICE)
criterion = smp.losses.DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            torch.save(model.state_dict(), "base_model.pt")
            print("Model Saved!")
            best_valid_loss = valid_loss

        log_file.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}\n")
        print("\n")

print(f"Training log saved to {LOG_FILE_PATH}")
plot_losses(train_losses, valid_losses, "Training Loss on Cityscapes Dataset", "base_loss_fig.png")
