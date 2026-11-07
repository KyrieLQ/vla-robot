"""Proposed M7a: two16-token visual grids; language/state/action mechanism unchanged."""
from dataclasses import dataclass
import torch
from torch import nn
from .attention_model_v1 import AttentionConfig,AttentionMiniVLA

@dataclass(frozen=True)
class SpatialTokensConfig(AttentionConfig):
    grid_size:int=4

class SpatialTokensMiniVLA(AttentionMiniVLA):
    def __init__(self,config,initialize_pretrained=False,pretrained_path=None):
        super().__init__(config,initialize_pretrained,pretrained_path)
        # Delete unused dense image projection parameters; avoid trainable dead branches.
        del self.vision.projection,self.visual_token
        self.grid_projection=nn.Linear(512,config.token_dimension)
        self.spatial_position=nn.Parameter(torch.empty(config.grid_size**2,config.token_dimension))
        nn.init.normal_(self.spatial_position,std=.02)

    def observation_tokens(self,*,image,wrist_image,state,token_ids,language_mask):
        b=image.shape[0];g=self.config.grid_size
        if image.shape!=(b,3,self.config.image_size,self.config.image_size) or wrist_image.shape!=image.shape:raise ValueError('Image shape')
        if state.shape!=(b,self.config.state_dim):raise ValueError('State shape')
        if token_ids.shape!=language_mask.shape or not language_mask.any(1).all():raise ValueError('Language mask')
        features=self.vision.backbone(self.vision.standardize(torch.cat([image,wrist_image],0)))
        if features.shape[2:]!=(g,g):raise ValueError('Fixed4x4 feature grid expected')
        visual=self.grid_projection(features.flatten(2).transpose(1,2))+self.spatial_position
        mask=language_mask.unsqueeze(-1).to(visual.dtype)
        text=self.language_token((self.language(token_ids)*mask).sum(1)/mask.sum(1))+self.token_type[2]
        robot=self.state_token(self.proprioception(state))+self.token_type[3]
        return torch.cat([visual[:b]+self.token_type[0],visual[b:]+self.token_type[1],text[:,None],robot[:,None]],1)

    def forward(self,**observations):
        tokens=self.observation_tokens(**observations)
        memory=self.encoder(tokens)
        queries=self.action_queries.unsqueeze(0).expand(tokens.shape[0],-1,-1)
        return self.action_output(self.decoder(queries,memory))
