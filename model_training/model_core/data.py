import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(img_size=(256, 512)):
    return A.Compose([
        A.Resize(*img_size),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class CityscapesDataset(Cityscapes):
    def __init__(self, root, split="train", mode="fine", target_type="semantic", transform=None):
        super().__init__(root, split=split, mode=mode, target_type=target_type)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        targets = [
            self._load_json(target) if t == "polygon" else Image.open(target)
            for t, target in zip(self.target_type, self.targets[index])
        ]
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(target))
            return augmented["image"], augmented["mask"]
        return image, target



class MapillaryDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_folder, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            return augmented["image"], augmented["mask"]
        return image, mask
