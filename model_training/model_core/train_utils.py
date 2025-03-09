import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, criterion, device, encode_fn):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(data_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # Use the provided encoding function to preprocess the masks
        masks_encoded = encode_fn(masks)
        loss = criterion(outputs, masks_encoded.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_fn(data_loader, model, criterion, device, encode_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            masks_encoded = encode_fn(masks)
            loss = criterion(outputs, masks_encoded.long())
            total_loss += loss.item()
    return total_loss / len(data_loader)
