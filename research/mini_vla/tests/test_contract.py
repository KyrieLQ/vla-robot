import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mini_vla.data import (DemonstrationDataset, action_window, build_vocabulary,
                           denormalize, encode_instruction, fit_normalization,
                           image_tensor, normalize, split_episodes)
from mini_vla.model import masked_mse


class DataContractTests(unittest.TestCase):
    def test_split_uses_episodes_not_frames_and_preserves_tasks(self):
        m = {'tasks': [{'task_id': 0, 'language': 'pick bowl'}, {'task_id': 1, 'language': 'close drawer'}],
             'episodes': [{'episode_index': i * 7 + 10, 'tasks': ['pick bowl' if i < 10 else 'close drawer'], 'length': 3}
                          for i in range(20)]}
        a = split_episodes(m, 42, 0.1)
        b = split_episodes({**m, 'episodes': list(reversed(m['episodes']))}, 42, 0.1)
        self.assertEqual(a, b)
        self.assertEqual(len(a['validation']), 2)
        self.assertFalse({e['episode_id'] for e in a['train']} & {e['episode_id'] for e in a['validation']})
        self.assertEqual({e['task_id'] for e in a['validation']}, {0, 1})

    def test_training_vocabulary_and_stats_have_no_validation_input(self):
        train = [{'instruction': 'pick bowl'}]
        vocab = build_vocabulary(train)
        ids, mask = encode_instruction('unseen bowl', vocab)
        self.assertEqual(ids[0].item(), 1)
        self.assertEqual(mask.sum().item(), 2)
        values = np.array([[1, 5], [3, 5]], dtype=np.float32)
        stats = fit_normalization({'state': values})['state']
        self.assertEqual(stats['mean'], [2, 5])
        self.assertEqual(stats['scale'], [1, 1])
        torch.testing.assert_close(denormalize(normalize(values, stats), stats), torch.tensor(values))

    def test_window_padding_and_masked_loss_ignore_invalid_targets(self):
        actions = np.arange(21, dtype=np.float32).reshape(3, 7)
        window, mask = action_window(actions, 2, 3)
        np.testing.assert_array_equal(window, np.repeat(actions[-1:], 3, axis=0))
        np.testing.assert_array_equal(mask, [True, False, False])
        prediction = torch.zeros(1, 3, 7, requires_grad=True)
        target = torch.ones_like(prediction)
        target[:, 1:] = float('nan')
        loss = masked_mse(prediction, target, torch.tensor([[True, False, False]]))
        self.assertEqual(loss.item(), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[:, 1:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            masked_mse(prediction, target, torch.zeros(1, 3, dtype=torch.bool))

    def test_rgb_channels_and_orientation_are_preserved(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]
        image[-1, -1] = [0, 255, 0]
        tensor = image_tensor(image, size=4)
        self.assertEqual(tensor[:, 0, 0].tolist(), [1, 0, 0])
        self.assertEqual(tensor[:, -1, -1].tolist(), [0, 1, 0])
        with self.assertRaises(ValueError):
            image_tensor(image.astype(np.float32))

    def test_noncontiguous_episode_ids_do_not_cross_boundaries(self):
        base = Path('tmp')
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mini_vla_test_', dir=base) as tmp:
            root = Path(tmp)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps({'chunks_size': 1000, 'data_path': 'episode_{episode_index}.parquet'}))
            buffer = io.BytesIO()
            Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(buffer, format='PNG')
            image = {'bytes': buffer.getvalue(), 'path': 'test.png'}
            episodes = []
            for eid, marker in [(17, 1), (902, 9)]:
                rows = [{'image': image, 'wrist_image': image, 'state': [0.] * 8,
                         'actions': [float(marker)] * 7, 'episode_index': eid, 'frame_index': i} for i in range(2)]
                pq.write_table(pa.Table.from_pylist(rows), root / f'episode_{eid}.parquet')
                episodes.append({'episode_id': eid, 'length': 2, 'task_id': 0, 'instruction': 'pick bowl'})
            norm = {'state': {'mean': [0.] * 8, 'scale': [1.] * 8},
                    'actions': {'mean': [0.] * 7, 'scale': [1.] * 7}}
            ds = DemonstrationDataset(root, episodes, build_vocabulary(episodes), norm, horizon=3, cache_episodes=1)
            end, next_start = ds[1], ds[2]
            self.assertEqual(end['episode_id'], 17)
            self.assertEqual(end['target_actions'].unique().tolist(), [1])
            self.assertEqual(end['action_mask'].tolist(), [True, False, False])
            self.assertEqual(next_start['episode_id'], 902)
            self.assertEqual(next_start['target_actions'].unique().tolist(), [9])
            self.assertEqual(len(ds._tables), 1)
            with self.assertRaises(IndexError):
                ds[len(ds)]


if __name__ == '__main__':
    unittest.main()
