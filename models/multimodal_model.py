import torch
import torch.nn as nn
from .vision_module import VisionModule
from .text_module import TextModule


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super(MultimodalModel, self).__init__()
        self.vision_module = VisionModule(config)
        self.text_module = TextModule(config)
        self.llm = nn.Linear(config['llm']['input_dim'], config['llm']['output_dim'])

    def forward(self, images, biomarkers):
        image_token = self.vision_module(images)
        text_token = self.text_module(biomarkers)
        combined_token = torch.cat([image_token, text_token], dim=1)
        output = self.llm(combined_token)
        return output
