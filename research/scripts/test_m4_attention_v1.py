import unittest
from pathlib import Path
import torch
from mini_vla.attention_model_v1 import AttentionMiniVLA,AttentionConfig
from mini_vla.visual_model_v1 import VisualMiniVLA,VisualConfig
from mini_vla.model import masked_mse

class AttentionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(42)
        cls.model=AttentionMiniVLA(AttentionConfig(28,horizon=10));cls.model.eval()
        cls.x=dict(image=torch.rand(2,3,128,128),wrist_image=torch.rand(2,3,128,128),state=torch.rand(2,8),token_ids=torch.tensor([[2,3,0],[4,5,0]]),language_mask=torch.tensor([[True,True,False],[True,True,False]]))
    def test_shape_padding_and_observation_dependence(self):
        with torch.no_grad():
            a=self.model(**self.x);self.assertEqual(a.shape,(2,10,7))
            x=dict(self.x);x['token_ids']=x['token_ids'].clone();x['token_ids'][:,-1]=17
            torch.testing.assert_close(a,self.model(**x),rtol=0,atol=0)
            x=dict(self.x);x['state']=x['state']+1
            self.assertGreater(float((a-self.model(**x)).abs().max()),1e-7)
    def test_gradient_and_no_future_input(self):
        self.model.zero_grad();a=self.model(**self.x);mask=torch.ones(2,10,dtype=torch.bool);mask[:,-2:]=False
        y=torch.zeros_like(a);y[:,-2:]=float('nan');loss=masked_mse(a,y,mask);self.assertTrue(torch.isfinite(loss));loss.backward()
        for name in ['vision.backbone.0.weight','encoder.layers.0.self_attn.in_proj_weight','decoder.layers.0.multihead_attn.in_proj_weight','action_queries','language.weight']:
            g=dict(self.model.named_parameters())[name].grad;self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)
        with self.assertRaises(TypeError):self.model(**self.x,future_actions=y)
    def test_shared_initialization(self):
        torch.manual_seed(42);base=VisualMiniVLA(VisualConfig(28,horizon=10,vision_variant='resnet_pretrained'))
        for k,t in base.state_dict().items():
            if k.startswith(('vision.','language.','proprioception.')):torch.testing.assert_close(t,self.model.state_dict()[k],rtol=0,atol=0)
    def test_serializable_configuration(self):
        cfg=AttentionConfig(**self.model.configuration());copy=AttentionMiniVLA(cfg);copy.load_state_dict(self.model.state_dict());copy.eval()
        with torch.no_grad():torch.testing.assert_close(copy(**self.x),self.model(**self.x),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
