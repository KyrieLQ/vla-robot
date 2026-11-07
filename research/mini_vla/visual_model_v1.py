"""M3 controls: input standardization, ResNet18 architecture, ImageNet initialization."""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18
from .model import ModelConfig
from .chunk_model_v1 import ChunkMiniVLA

@dataclass(frozen=True)
class VisualConfig(ModelConfig):
    vision_variant: str = 'tiny_norm'

class RGBStandardize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def forward(self,image):return (image-self.mean)/self.std

class VisualEncoder(nn.Module):
    def __init__(self,backbone,projection):
        super().__init__();self.standardize=RGBStandardize();self.backbone=backbone;self.projection=projection
    def forward(self,image):return self.projection(self.backbone(self.standardize(image)))

class VisualMiniVLA(ChunkMiniVLA):
    def __init__(self,config,initialize_pretrained=False,pretrained_path=None):
        super().__init__(config)
        if config.vision_variant=='tiny_norm':
            self.vision=VisualEncoder(self.vision,nn.Identity())
        elif config.vision_variant in ['resnet_random','resnet_pretrained']:
            network=resnet18(weights=None)  # Never downloads implicitly.
            if config.vision_variant=='resnet_pretrained' and initialize_pretrained:
                if pretrained_path is None:raise ValueError('Explicit checked local weights required')
                state=torch.load(Path(pretrained_path),map_location='cpu',weights_only=True)
                network.load_state_dict(state,strict=True)
                torch.testing.assert_close(network.conv1.weight,state['conv1.weight'],atol=0,rtol=0)
            backbone=nn.Sequential(*list(network.children())[:-2])
            projection=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512*4*4,256),nn.ReLU())
            self.vision=VisualEncoder(backbone,projection)
        else:raise ValueError('Unknown visual variant')
