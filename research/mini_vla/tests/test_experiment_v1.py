import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import RandomSampler
from mini_vla.experiment_v1 import CachedData,inputs
from mini_vla.model import MiniVLA,ModelConfig,masked_mse

class ExperimentTests(unittest.TestCase):
    def test_cache_keeps_camera_order_ids_and_normalized_labels(self):
        with tempfile.TemporaryDirectory(dir='tmp',prefix='m1_cache_test_') as tmp:
            directory=Path(tmp)/'train';directory.mkdir()
            images=np.zeros((2,2,3,128,128),dtype='uint8');images[:,0,0]=255;images[:,1,1]=128
            values={'images':images,'state':np.ones((2,8),dtype='float32'),'actions':np.full((2,1,7),.25,dtype='float32'),
                    'token_ids':np.ones((2,32),dtype='int64'),'language_mask':np.ones((2,32),dtype='bool'),
                    'episode_id':np.array([17,902]),'task_id':np.array([0,9])}
            for k,v in values.items():np.save(directory/(k+'.npy'),v)
            data=CachedData(tmp,'train');batch={k:torch.stack([data[0][k],data[1][k]]) for k in data[0]}
            x=inputs(batch,torch.device('cpu'))
            self.assertEqual(x['image'][:,0].unique().tolist(),[1.0])
            self.assertAlmostEqual(x['wrist_image'][:,1].mean().item(),128/255,places=6)
            self.assertEqual(batch['episode_id'].tolist(),[17,902])
            self.assertEqual(batch['actions'].unique().tolist(),[.25])
            self.assertNotIn('actions',x)

    def test_completed_epoch_resume_uses_same_frame_order(self):
        generator=torch.Generator();sampler=RandomSampler(range(100),generator=generator)
        generator.manual_seed(43);list(sampler)
        generator.manual_seed(44);continuous=list(sampler)
        restored=RandomSampler(range(100),generator=torch.Generator().manual_seed(44))
        self.assertEqual(continuous,list(restored));self.assertEqual(len(set(continuous)),100)

    def test_optimizer_state_resume_preserves_next_update(self):
        torch.set_num_threads(4);torch.manual_seed(123)
        model=MiniVLA(ModelConfig(28));opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
        x=dict(image=torch.zeros(1,3,128,128),wrist_image=torch.ones(1,3,128,128),state=torch.ones(1,8),
               token_ids=torch.ones(1,32,dtype=torch.long),language_mask=torch.ones(1,32,dtype=torch.bool))
        target=torch.ones(1,1,7);mask=torch.ones(1,1,dtype=torch.bool)
        def step(m,o):
            o.zero_grad();loss=masked_mse(m(**x),target,mask);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
        step(model,opt)
        with tempfile.TemporaryDirectory(dir='tmp',prefix='m1_resume_test_') as tmp:
            path=Path(tmp)/'checkpoint.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},path)
            restored=MiniVLA(ModelConfig(28));other=torch.optim.AdamW(restored.parameters(),lr=3e-4)
            state=torch.load(path,weights_only=False);restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
            step(model,opt);step(restored,other)
            for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
