import unittest
import torch
from mini_vla.visual_model_v1 import VisualMiniVLA,VisualConfig,RGBStandardize
from mini_vla.model import masked_mse
from mini_vla.experiment_v1 import ROOT

class VisualTests(unittest.TestCase):
    def test_standardization_is_fixed_and_has_known_answer(self):
        norm=RGBStandardize();x=torch.tensor([.485,.456,.406]).reshape(1,3,1,1).expand(2,3,4,4)
        torch.testing.assert_close(norm(x),torch.zeros_like(x),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in norm.parameters()),0)

    def test_tiny_control_preserves_all_initial_parameters(self):
        torch.manual_seed(42);model=VisualMiniVLA(VisualConfig(28,horizon=10,vision_variant='tiny_norm'))
        original=torch.load(ROOT/'results/mini_vla_m2_20261019_run1/training/initial.pt',map_location='cpu',weights_only=False)['state_dict']
        for key,value in original.items():
            name=key.replace('vision.','vision.backbone.',1) if key.startswith('vision.') else key
            torch.testing.assert_close(model.state_dict()[name],value,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),794726)

    def test_resnet_shape_gradients_and_matched_nonvisual_initialization(self):
        torch.set_num_threads(4);torch.manual_seed(42)
        model=VisualMiniVLA(VisualConfig(28,horizon=10,vision_variant='resnet_random'))
        original=torch.load(ROOT/'results/mini_vla_m2_20261019_run1/training/initial.pt',map_location='cpu',weights_only=False)['state_dict']
        for key,value in original.items():
            if not key.startswith('vision.'):torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),13781894)
        x=dict(image=torch.rand(1,3,128,128),wrist_image=torch.rand(1,3,128,128),state=torch.ones(1,8),
               token_ids=torch.ones(1,32,dtype=torch.long),language_mask=torch.ones(1,32,dtype=torch.bool))
        output=model(**x);self.assertEqual(tuple(output.shape),(1,10,7))
        masked_mse(output,torch.ones_like(output),torch.ones(1,10,dtype=torch.bool)).backward()
        self.assertGreater(float(model.vision.backbone[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.requires_grad for p in model.vision.backbone.parameters()))
        with self.assertRaises(TypeError):model.predict(**x,actions=torch.zeros(1,10,7))
if __name__=='__main__':unittest.main()
