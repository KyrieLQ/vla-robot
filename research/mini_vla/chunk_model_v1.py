"""M2: retain the M1 observation trunk and predict K actions in parallel."""
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from .model import MiniVLA
from .experiment_v1 import CachedData

class ChunkMiniVLA(MiniVLA):
    def __init__(self,config):
        super().__init__(replace(config,horizon=1))
        if config.horizon<1:raise ValueError('Invalid horizon')
        self.config=config
        if config.horizon!=1:
            self.action_head[-1]=nn.Linear(128,config.action_dim*config.horizon)

    def forward(self,**observations):
        output=super().forward(**observations)
        return output.reshape(output.shape[0],self.config.horizon,self.config.action_dim)

class ChunkCachedData(CachedData):
    def __init__(self,cache,part,horizon=10):
        super().__init__(cache,part);self.horizon=horizon
        ids=self.arrays['episode_id']
        self.ends=np.r_[np.flatnonzero(ids[1:]!=ids[:-1]),len(ids)-1]

    def window_indices(self,index):
        if index<0 or index>=len(self):raise IndexError(index)
        end=self.ends[np.searchsorted(self.ends,index,side='left')]
        wanted=np.arange(index,index+self.horizon)
        return np.minimum(wanted,end),wanted<=end

    def __getitem__(self,index):
        batch=super().__getitem__(index);indices,mask=self.window_indices(index)
        batch['actions']=torch.tensor(np.array(self.arrays['actions'][indices,0],copy=True))
        batch['action_mask']=torch.from_numpy(mask)
        return batch
