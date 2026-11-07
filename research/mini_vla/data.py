"""Episode-safe data access. No LeRobot import, download, or simulator startup."""
from __future__ import annotations

import bisect
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict) -> None:
    """A rerun may verify an existing artifact, never silently change it."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


def split_episodes(manifest: dict, seed: int, fraction: float) -> dict:
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1')
    result = {'seed': seed, 'validation_fraction': fraction, 'train': [], 'validation': []}
    seen = set()
    for task in manifest['tasks']:
        episodes = [e for e in manifest['episodes'] if e['tasks'] == [task['language']]]
        if len(episodes) < 2:
            raise ValueError('Each task needs at least two complete demonstrations')
        episodes.sort(key=lambda e: hashlib.sha256(f"{seed}:{e['episode_index']}".encode()).hexdigest())
        n_val = min(len(episodes) - 1, math.ceil(len(episodes) * fraction))
        for i, e in enumerate(episodes):
            eid = e['episode_index']
            if eid in seen:
                raise ValueError('Repeated episode id')
            seen.add(eid)
            result['validation' if i < n_val else 'train'].append({
                'episode_id': eid, 'length': e['length'],
                'task_id': task['task_id'], 'instruction': task['language'],
            })
    if len(seen) != len(manifest['episodes']):
        raise ValueError('Manifest contains unmapped episodes')
    for name in ['train', 'validation']:
        result[name].sort(key=lambda e: e['episode_id'])
    return result


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace('_', ' '))


def build_vocabulary(train_episodes: list[dict]) -> dict[str, int]:
    tokens = sorted({w for e in train_episodes for w in words(e['instruction'])})
    return {'<PAD>': 0, '<UNK>': 1, **{w: i + 2 for i, w in enumerate(tokens)}}


def encode_instruction(text: str, vocabulary: dict, length: int = 32):
    tokens = words(text)
    if not tokens or len(tokens) > length:
        raise ValueError('Instruction is empty or exceeds the fixed length')
    ids = torch.zeros(length, dtype=torch.long)
    mask = torch.zeros(length, dtype=torch.bool)
    ids[:len(tokens)] = torch.tensor([vocabulary.get(t, 1) for t in tokens])
    mask[:len(tokens)] = True
    return ids, mask


def fit_normalization(train_arrays: dict[str, np.ndarray]) -> dict:
    result = {}
    for key, value in train_arrays.items():
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
            raise ValueError('Expected finite, nonempty training vectors')
        std = value.std(axis=0, ddof=0)
        result[key] = {'mean': value.mean(axis=0).tolist(),
                       'scale': np.where(std < 1e-6, 1.0, std).tolist(),
                       'constant_dimensions': np.flatnonzero(std < 1e-6).tolist()}
    return result


def normalize(value, stats: dict) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return (tensor - tensor.new_tensor(stats['mean'])) / tensor.new_tensor(stats['scale'])


def denormalize(value: torch.Tensor, stats: dict) -> torch.Tensor:
    return value * value.new_tensor(stats['scale']) + value.new_tensor(stats['mean'])


def image_tensor(rgb: np.ndarray, size: int = 128) -> torch.Tensor:
    """Already oriented RGB -> CHW. Training and inference use the same resize."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB')
    resized = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    array = np.array(resized, copy=True).transpose(2, 0, 1).copy()
    return torch.from_numpy(array).float().div_(255.0)


def decode_image(cell: dict, size: int = 128) -> torch.Tensor:
    if not cell.get('bytes'):
        raise ValueError('This adapter expects embedded parquet image bytes')
    with Image.open(io.BytesIO(cell['bytes'])) as im:
        return image_tensor(np.asarray(im.convert('RGB')), size)


def action_window(actions: np.ndarray, frame: int, horizon: int):
    """Padding repeats the last action but carries zero loss weight."""
    if horizon < 1 or frame < 0 or frame >= len(actions):
        raise ValueError('Invalid window')
    indices = np.arange(frame, frame + horizon)
    mask = indices < len(actions)
    return actions[np.minimum(indices, len(actions) - 1)], mask


def episode_path(dataset_root: Path, info: dict, episode_id: int) -> Path:
    return dataset_root / info['data_path'].format(
        episode_chunk=episode_id // info['chunks_size'], episode_index=episode_id)


class DemonstrationDataset(Dataset):
    """Indexes actual episode IDs; bounded cache avoids loading all image data."""

    def __init__(self, dataset_root, episodes, vocabulary, normalization,
                 image_size=128, language_length=32, horizon=1, cache_episodes=2):
        self.root = Path(dataset_root)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.episodes = episodes
        self.vocabulary, self.normalization = vocabulary, normalization
        self.image_size, self.language_length = image_size, language_length
        self.horizon, self.cache_episodes = horizon, cache_episodes
        if cache_episodes < 1 or horizon < 1 or not episodes:
            raise ValueError('Empty dataset or invalid cache/window')
        self.ends = np.cumsum([e['length'] for e in episodes]).tolist()
        self._tables = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        start = self.ends[episode - 1] if episode else 0
        return self.episodes[episode], index - start

    def _table(self, e):
        eid = e['episode_id']
        if eid not in self._tables:
            table = pq.read_table(episode_path(self.root, self.info, eid),
                                  columns=['image', 'wrist_image', 'state', 'actions', 'episode_index', 'frame_index'],
                                  use_threads=False)
            if (table.num_rows != e['length'] or
                set(table['episode_index'].to_pylist()) != {eid} or
                table['frame_index'].to_pylist() != list(range(e['length']))):
                raise ValueError('Episode identity or ordering mismatch')
            self._tables[eid] = table
            if len(self._tables) > self.cache_episodes:
                self._tables.popitem(last=False)
        self._tables.move_to_end(eid)
        return self._tables[eid]

    def __getitem__(self, index):
        e, frame = self.locate(index)
        table = self._table(e)
        row = table.slice(frame, 1).to_pylist()[0]
        actions = np.asarray(table['actions'].to_pylist(), dtype=np.float32)
        target, mask = action_window(actions, frame, self.horizon)
        ids, language_mask = encode_instruction(e['instruction'], self.vocabulary, self.language_length)
        # Keep observations separate from targets: infer has no label argument.
        return {
            'observations': {
                'image': decode_image(row['image'], self.image_size),
                'wrist_image': decode_image(row['wrist_image'], self.image_size),
                'state': normalize(row['state'], self.normalization['state']),
                'token_ids': ids, 'language_mask': language_mask,
            },
            'target_actions': normalize(target, self.normalization['actions']),
            'action_mask': torch.from_numpy(mask),
            'episode_id': e['episode_id'], 'frame_index': frame, 'task_id': e['task_id'],
        }
