"""CPU-only checks for a proposed model; no optimizer or simulation."""
import json,time,unittest
from pathlib import Path
import torch
from mini_vla.attention_model_v1 import AttentionMiniVLA,AttentionConfig
from mini_vla.spatial_tokens_model_v1 import SpatialTokensMiniVLA,SpatialTokensConfig

class SpatialTokenChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(42);cls.model=SpatialTokensMiniVLA(SpatialTokensConfig(71,horizon=10));cls.model.eval()
        cls.x=dict(image=torch.rand(2,3,128,128),wrist_image=torch.rand(2,3,128,128),state=torch.rand(2,8),token_ids=torch.tensor([[2,3,0],[4,5,0]]),language_mask=torch.tensor([[True,True,False],[True,True,False]]))
    def test_shape_and_padding(self):
        with torch.no_grad():
            self.assertEqual(self.model.observation_tokens(**self.x).shape,(2,34,128));pred=self.model(**self.x);self.assertEqual(pred.shape,(2,10,7))
            x=dict(self.x);x['token_ids']=x['token_ids'].clone();x['token_ids'][:,-1]=33;torch.testing.assert_close(pred,self.model(**x),atol=0,rtol=0)
    def test_shared_initialization(self):
        torch.manual_seed(42);base=AttentionMiniVLA(AttentionConfig(71,horizon=10))
        state=self.model.state_dict()
        for k,t in base.state_dict().items():
            if k in state:torch.testing.assert_close(t,state[k],atol=0,rtol=0)
    def test_no_dead_parameters_and_no_future_label(self):
        self.model.zero_grad();self.model(**self.x).square().mean().backward()
        for name,p in self.model.named_parameters():self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(float(self.model.spatial_position.grad.abs().sum()),0)
        self.assertNotIn('vision.projection.2.weight',self.model.state_dict())
        with self.assertRaises(TypeError):self.model(**self.x,actions=torch.zeros(2,10,7))

if __name__=='__main__':
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(SpatialTokenChecks);result=unittest.TextTestRunner().run(suite)
    if result.wasSuccessful():
        report=dict(status='PASS',checks=result.testsRun,elapsed_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in SpatialTokenChecks.model.parameters()),visual_tokens=32,observation_tokens=34,optimizer_updates=0,simulation_episodes=0)
        path=Path(__file__).resolve().parents[1]/'reports/mini_vla_m7a_cpu_proposal_checks_20261020.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
