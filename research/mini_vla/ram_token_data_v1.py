"""Full BF16 adapted tokens in process RAM; old disk cache supplies layout only."""
import bisect
import numpy as np
import torch
from .vlm_cached_data_v2 import VLMCacheData
from .vlm_frontend_v1 import bits_bf16

class RamTokenData(VLMCacheData):
    def __init__(self, layout_cache, numeric_cache, part, tokens):
        super().__init__(layout_cache, numeric_cache, part)
        self.tokens=tokens
        for e in self.index:
            a=tokens[e['episode_id']]
            assert a.dtype==np.uint16 and a.shape==(e['end']-e['start'],e['tokens'],960)
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        e=self.index[bisect.bisect_right(self.ends,index)];frame=index-e['start']
        hidden=bits_bf16(self.tokens[e['episode_id']][frame]);target,mask=self.window_indices(index)
        return dict(hidden=hidden,hidden_mask=torch.ones(len(hidden),dtype=torch.bool),state=torch.from_numpy(np.array(self.numeric['state'][index])),actions=torch.from_numpy(np.array(self.numeric['actions'][target,0])),action_mask=torch.from_numpy(mask),episode_id=e['episode_id'],frame_index=frame,task_id=int(self.numeric['task_id'][index]))
