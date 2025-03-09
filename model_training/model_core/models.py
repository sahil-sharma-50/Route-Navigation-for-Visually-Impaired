import torch.nn as nn
import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):
    def __init__(self, encoder="resnet34", num_classes=8):
        super(SegmentationModel, self).__init__()
        self.layer = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.layer(x)
