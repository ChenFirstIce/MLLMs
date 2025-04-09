import torch
import torch.nn as nn
from transformers import TransformerEncoder, TransformerEncoderLayer
from utils.moe_utils import MoEDecoupling


class TextModule(nn.Module):
    def __init__(self, config):
        super(TextModule, self).__init__()
        self.encoder = nn.Linear(config['text']['input_dim'], config['text']['d_model'])
        encoder_layer = TransformerEncoderLayer(d_model=config['text']['d_model'], nhead=config['text']['nhead'])
        self.transformer = TransformerEncoder(encoder_layer, num_layers=config['text']['num_layers'])
        self.moe_decoupling = MoEDecoupling(config)

    def forward(self, biomarkers):
        encoded_biomarkers = self.encoder(biomarkers)
        decoupled_output = self.transformer(encoded_biomarkers.unsqueeze(1))
        text_token = self.moe_decoupling(decoupled_output.squeeze(1))
        return text_token
