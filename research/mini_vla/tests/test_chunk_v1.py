import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from mini_vla.chunk_model_v1 import ChunkMiniVLA,ChunkCachedData
from mini_vla.model import ModelConfig,masked_mse
from mini_vla.experiment_v1 import ROOT,inputs

class ChunkTests(unittest.TestCase):
    def test_trunk_matches_m1_initial_and_output_is_ten_actions(self):
        torch.set_num_threads(4);torch.manual_seed(42)
        model=ChunkMiniVLA(ModelConfig(28,horizon=10))
        reference=torch.load(ROOT/'results/mini_vla_m1_20261019_run1/training/initial.pt',map_location='cpu',weights_only=False)
        for key,value in model.state_dict().items():
            if not key.startswith('action_head.2'):torch.testing.assert_close(value,reference['state_dict'][key],atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),794726)
        x=dict(image=torch.zeros(2,3,128,128),wrist_image=torch.ones(2,3,128,128),state=torch.ones(2,8),
               token_ids=torch.ones(2,32,dtype=torch.long),language_mask=torch.ones(2,32,dtype=torch.bool))
        output=model(**x);self.assertEqual(tuple(output.shape),(2,10,7))
        masked_mse(output,torch.ones_like(output),torch.ones(2,10,dtype=torch.bool)).backward()
        gradient=model.action_head[-1].weight.grad.reshape(10,7,128)
        self.assertTrue(torch.isfinite(gradient).all());self.assertTrue((gradient.abs().sum(dim=(1,2))>0).all())
        with self.assertRaises(TypeError):model.predict(**x,actions=torch.zeros(2,10,7))

    def test_real_episode_boundaries_and_target_counts(self):
        for part,expected in [('train',452540),('validation',57720)]:
            ds=ChunkCachedData(ROOT/'datasets/mini_vla_128_20261019_v1',part)
            ids=ds.arrays['episode_id'];total=0
            for index in range(len(ds)):
                positions,mask=ds.window_indices(index)
                self.assertTrue(np.all(ids[positions]==ids[index]))
                total+=int(mask.sum())
            self.assertEqual(total,expected)
            end=int(ds.ends[0]);sample=ds[end]
            self.assertEqual(sample['action_mask'].tolist(),[True]+[False]*9)
            expected_action=torch.tensor(np.array(ds.arrays['actions'][end,0]))
            torch.testing.assert_close(sample['actions'],expected_action.repeat(10,1))
            self.assertNotEqual(int(ids[end]),int(ids[end+1]))

    def test_queue_uses_prefix_without_peeking_at_new_observations(self):
        from collections import deque
        for h in [1,5]:
            queue=deque();executed=[];chunks=[];calls=[]
            for step in range(13):
                if not queue:
                    chunk=np.arange(70).reshape(10,7)+step*100;chunks.append(chunk);calls.append(step);queue.extend(chunk[:h])
                executed.append(queue.popleft())
            np.testing.assert_array_equal(calls,np.arange(0,13,h))
            np.testing.assert_array_equal(executed,np.asarray(chunks)[:,:h].reshape(-1,7)[:13])
if __name__=='__main__':unittest.main()
