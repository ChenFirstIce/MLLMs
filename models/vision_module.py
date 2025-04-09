import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils.moe_utils import MoEAlignment


class VisionModule(nn.Module):
    def __init__(self, config):
        super(VisionModule, self).__init__()
        self.encoder = resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.vit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config['vision']['d_model'], nhead=config['vision']['nhead']),
            num_layers=config['vision']['num_layers']
        )
        self.moe_alignment = MoEAlignment(config)

    def forward(self, images):
        encoded_images = self.encoder(images)
        dynamic_token = torch.randn(encoded_images.size(0), 1, encoded_images.size(-1))
        input_with_token = torch.cat([dynamic_token, encoded_images], dim=1)
        output = self.vit(input_with_token)
        image_token = self.moe_alignment(output)
        return image_token
