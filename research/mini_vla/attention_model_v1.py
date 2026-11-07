"""M4: four observation tokens, Transformer fusion and ten learned action queries.

Language remains masked-mean pooled to isolate the fusion/generation block.
No future labels, CVAE posterior, diffusion noise or temporal ensemble is used.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .visual_model_v1 import VisualConfig,VisualMiniVLA


@dataclass(frozen=True)
class AttentionConfig(VisualConfig):
    vision_variant: str = 'resnet_pretrained'
    token_dimension: int = 128
    attention_heads: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 1


class AttentionMiniVLA(VisualMiniVLA):
    def __init__(self,config,initialize_pretrained=False,pretrained_path=None):
        if config.vision_variant!='resnet_pretrained':raise ValueError('M4 uses matched pretrained ResNet18')
        super().__init__(config,initialize_pretrained,pretrained_path)
        del self.fusion,self.action_head
        d=config.token_dimension
        self.visual_token=nn.Linear(256,d)
        self.language_token=nn.Linear(64,d)
        self.state_token=nn.Linear(64,d)
        self.token_type=nn.Parameter(torch.empty(4,d))
        self.action_queries=nn.Parameter(torch.empty(config.horizon,d))
        nn.init.normal_(self.token_type,std=.02);nn.init.normal_(self.action_queries,std=.02)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,config.attention_heads,2*d,dropout=0.,activation='gelu',batch_first=True),config.encoder_layers,enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d,config.attention_heads,2*d,dropout=0.,activation='gelu',batch_first=True),config.decoder_layers)
        self.action_output=nn.Linear(d,config.action_dim)

    def forward(self,*,image,wrist_image,state,token_ids,language_mask):
        b=image.shape[0]
        if image.shape!=(b,3,self.config.image_size,self.config.image_size) or wrist_image.shape!=image.shape:raise ValueError('Two BCHW images required')
        if state.shape!=(b,self.config.state_dim):raise ValueError('State shape')
        if token_ids.shape!=language_mask.shape or token_ids.ndim!=2 or not language_mask.any(1).all():raise ValueError('Nonempty language mask required')
        visual=self.vision(torch.cat([image,wrist_image],0));mask=language_mask.unsqueeze(-1).to(visual.dtype)
        text=(self.language(token_ids)*mask).sum(1)/mask.sum(1)
        tokens=torch.stack([self.visual_token(visual[:b]),self.visual_token(visual[b:]),self.language_token(text),self.state_token(self.proprioception(state))],1)
        memory=self.encoder(tokens+self.token_type.unsqueeze(0))
        queries=self.action_queries.unsqueeze(0).expand(b,-1,-1)
        return self.action_output(self.decoder(queries,memory))
