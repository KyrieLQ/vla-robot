"""All-frame episode-safe data, preserving the audited official preprocessing."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib,json
import numpy as np
import jax
import torch
from .pi05_four_suite_v1 import R,RawTrainingInputs

class FrameIndex:
    def __init__(self,entries):
        self.entries=entries;self.ends=[];self.starts={};total=0
        for row in entries:
            assert row['episode_id'] not in self.starts and row['length']>0
            self.starts[row['episode_id']]=total;total+=row['length'];self.ends.append(total)
        self.total=total
    def __len__(self):return self.total
    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<self.total:raise IndexError(index)
        row=self.entries[bisect_right(self.ends,index)]
        return dict(episode_id=row['episode_id'],frame_index=index-self.starts[row['episode_id']])
    def locate(self,eid,frame):
        index=self.starts[eid]+int(frame);row=self[index]
        assert row['episode_id']==eid and row['frame_index']==frame
        return index

class EpisodeCache(OrderedDict):
    def __init__(self):super().__init__();self.sizes={};self.bytes=0
    def __getitem__(self,key):
        value=super().__getitem__(key);self.move_to_end(key);return value
    def add(self,key,rows):
        size=sum(len(row['image']['bytes'])+len(row['wrist_image']['bytes'])+4096 for row in rows)
        if size>2*2**30:raise RuntimeError('Oneepisode exceeds boundedRAMcache')
        if key in self:self.bytes-=self.sizes.pop(key);super().__delitem__(key)
        super().__setitem__(key,rows);self.sizes[key]=size;self.bytes+=size
        while len(self)>64 or self.bytes>2*2**30:
            victim=next(iter(self));self.bytes-=self.sizes.pop(victim);super().__delitem__(victim)

class FullDataset(RawTrainingInputs):
    def __init__(self,plan,part):
        assert part in ('train','validation')
        super().__init__(plan)
        split=json.loads((R/plan['split']).read_text());self.part=part;self.index=FrameIndex(split[part])
        self.samples=self.index;self.episodes={row['episode_id']:row for row in split[part]}
        assert len(self.index)==(243000 if part=='train' else 30465)
    @property
    def table_cache(self):return self._episode_cache
    @table_cache.setter
    def table_cache(self,value):
        if not hasattr(self,'_episode_cache'):self._episode_cache=EpisodeCache()
        for key,rows in value.items():self._episode_cache.add(key,rows)
    def get(self,index):
        if not 0<=int(index)<len(self.index):raise IndexError(index)
        observation,actions,receipt=super().get(int(index))
        return observation,actions,dict(receipt,global_index=int(index))
    def get_batch(self,indices):
        items=[self.get(int(i)) for i in indices]
        batched=jax.tree.map(lambda *xs:np.stack(xs),*[item[:2] for item in items])
        return batched,[item[2] for item in items]

def epoch_order(epoch):
    assert 1<=epoch<=3
    order=np.random.default_rng(42+epoch).permutation(243000).astype('<i8')
    return order,hashlib.sha256(order.tobytes()).hexdigest()

def validation_noise(rows):
    result=[]
    for row in rows:
        seed=int.from_bytes(hashlib.sha256(('20261105:%d:%d'%(row['episode_id'],row['frame_index'])).encode()).digest()[:8],'little')%(2**63-1)
        result.append(torch.randn((10,32),generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy())
    return np.stack(result)
