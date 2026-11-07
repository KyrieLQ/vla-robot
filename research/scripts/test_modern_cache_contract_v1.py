"""Meaningful CPU failure controls before any full VLM cache/training launch."""
import json
import tempfile
import time
from pathlib import Path
import unittest

import numpy as np
import torch

from mini_vla.vlm_frontend_v1 import bf16_bits, bits_bf16
from mini_vla.vlm_cache_v1 import atomic_npy
from mini_vla.vlm_cached_data_v1 import VLMCacheData, collate_cached
from mini_vla.vlm_action_heads_v1 import RegressionHead, FlowHead, HeadConfig, flow_loss, example_seed


class Contracts(unittest.TestCase):
    def test_all_bf16_bit_patterns_survive_disk(self):
        bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        np.testing.assert_array_equal(bf16_bits(bits_bf16(bits)), bits)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"features.npy"
            atomic_npy(p, bits)
            np.testing.assert_array_equal(bf16_bits(bits_bf16(np.load(p))), bits)
        with self.assertRaises(ValueError):
            bf16_bits(torch.zeros(3, dtype=torch.float32))

    def test_episode_boundaries_and_variable_token_collation(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            cache, numeric = r/"cache", r/"numeric"
            (cache/"train").mkdir(parents=True)
            (numeric/"train").mkdir(parents=True)
            ids, index = [7, 7, 1003, 1003], []
            for eid, first, tokens in [(7, 0, 3), (1003, 2, 5)]:
                d = cache/"train"/str(eid)
                d.mkdir()
                x = torch.full((2, tokens, 960), float(eid), dtype=torch.bfloat16)
                atomic_npy(d/"hidden.npy", bf16_bits(x))
                index.append(dict(episode_id=eid, start=first, end=first+2,
                                  directory=f"train/{eid}", tokens=tokens))
            (cache/"train/index.json").write_text(json.dumps(index))
            (cache/"complete.json").write_text(json.dumps(dict(status="PASS", parts={"train": {"frames": 4}})))
            for k, v in dict(state=np.zeros((4, 8), np.float32),
                             actions=np.arange(28, dtype=np.float32).reshape(4, 1, 7),
                             episode_id=np.array(ids), task_id=np.array([0, 0, 1, 1])).items():
                atomic_npy(numeric/"train"/(k+".npy"), v)
            ds = VLMCacheData(cache, numeric, "train", max_open_episodes=1)
            idx, valid = ds.window_indices(1)
            np.testing.assert_array_equal(idx, np.ones(10, dtype=int))
            np.testing.assert_array_equal(valid, np.array([1]+[0]*9, dtype=bool))
            self.assertEqual(ds[2]["episode_id"], 1003)
            batch = collate_cached([ds[1], ds[2]])
            self.assertEqual(tuple(batch["hidden"].shape), (2, 5, 960))
            self.assertEqual(batch["hidden_mask"].sum(1).tolist(), [3, 5])
            self.assertEqual(batch["frame_index"].tolist(), [1, 0])
            self.assertLessEqual(len(ds.opened), 1)
            with self.assertRaises(IndexError):
                ds[4]

    def observations(self):
        torch.manual_seed(42)
        return dict(hidden=torch.randn(2, 5, 960),
                    hidden_mask=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
                    state=torch.randn(2, 8))

    def test_observation_padding_cannot_affect_regression(self):
        obs = self.observations()
        model = RegressionHead().eval()
        result = model(**obs)
        self.assertEqual(tuple(result.shape), (2, 10, 7))
        altered = dict(obs, hidden=obs["hidden"].clone())
        altered["hidden"][0, 3:] = float("nan")
        torch.testing.assert_close(result, model(**altered), rtol=0, atol=0)
        with self.assertRaises(TypeError):
            model(**obs, actions=torch.zeros(2, 10, 7))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_shared_initialization_and_flow_padding(self):
        torch.manual_seed(42)
        reg = RegressionHead()
        torch.manual_seed(42)
        flow = FlowHead()
        for k, v in reg.state_dict().items():
            torch.testing.assert_close(v, flow.state_dict()[k], rtol=0, atol=0)
        obs = self.observations()
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 2:] = False
        target, noise, time_value = torch.randn(2, 10, 7), torch.randn(2, 10, 7), torch.tensor([.2, .7])
        loss = flow_loss(flow, obs, target, mask, noise, time_value)
        target2 = target.clone()
        target2[0, 2:] = float("nan")
        torch.testing.assert_close(loss, flow_loss(flow, obs, target2, mask, noise, time_value), rtol=0, atol=0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters()))

    def test_euler_known_velocity_and_stable_per_example_seed(self):
        class ConstantVelocity(FlowHead):
            def velocity(self, *, noisy_action, **kwargs):
                return torch.ones_like(noisy_action)
        obs = self.observations()
        model = ConstantVelocity().eval()
        noise = torch.randn(2, 10, 7)
        torch.testing.assert_close(model.sample(**obs, noise=noise), noise+1, rtol=0, atol=2e-6)
        self.assertEqual(example_seed(7, 12), example_seed(7, 12))
        self.assertNotEqual(example_seed(7, 12), example_seed(7, 13))


if __name__ == "__main__":
    started = time.monotonic()
    torch.set_num_threads(4)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report = dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                  elapsed_seconds=time.monotonic()-started, gpu_operations=0, optimizer_updates=0,
                  cache_bitpatterns_tested=65536)
    (Path(__file__).resolve().parents[1]/"reports/modern_cache_cpu_contract_20261020.json").write_text(
        json.dumps(report, indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
