import torch
from torchvision import transforms
from PIL import Image
import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToPILImage

model_path = "road_scene_ImgSeg_model.pt"
input_images_folder = "Mapillary-Vistas-1000-sidewalks/testing/images"
output_masks_folder = "Mapillary-Output"
os.makedirs(output_masks_folder, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.arc = smp.DeepLabV3Plus(
            encoder_name='resnet34',
            in_channels=3,
            classes=1,
            activation=None
        )
        self.activation = nn.Sigmoid()

    def forward(self, images, masks=None):
        logits = self.arc(images)

        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2
        return self.activation(logits)


# Load the model
model = SegmentationModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Iterate through input images
for image_name in tqdm(os.listdir(input_images_folder), desc="Inferencing", unit="image"):
    image_path = os.path.join(input_images_folder, image_name)
    input_image = Image.open(image_path).convert("RGB")

    try:
        # Check if the image dimensions are divisible by 16
        if input_image.size[0] % 16 != 0 or input_image.size[1] % 16 != 0:
            raise RuntimeError(f"Image {image_name} dimensions not divisible by 16. Skipping.")

        input_tensor = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            output_mask = model(input_tensor)

        # Resize the output mask to the original size
        output_mask = torch.nn.functional.interpolate(output_mask, size=input_image.size[::-1], mode='bilinear', align_corners=False)

        output_mask_path = os.path.join(output_masks_folder, f"{image_name.split('.')[0]}.png")

        # Post-process and save the output mask
        output_mask = torch.clamp(output_mask, 0, 1)
        output_array = (output_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        output_image = ToPILImage()(output_array)
        output_image = output_image.convert('L')
        output_image.save(output_mask_path)

    except RuntimeError as e:
        print(e)
        # Skip to the next image in case of an exception

print("Inference completed. Predicted masks saved in:", output_masks_folder)
