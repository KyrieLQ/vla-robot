"""Freeze M0 data artifacts using existing local files only. Run from project root."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from .data import (DemonstrationDataset, build_vocabulary, episode_path,
                   fit_normalization, sha256, split_episodes, write_once)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='configs/mini_vla_m0_20261015_v1.json')
    args = parser.parse_args()
    start = time.monotonic()
    protocol_path = Path(args.protocol)
    p = json.loads(protocol_path.read_text())
    source = Path(p['source_manifest'])
    if sha256(source) != p['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    manifest = json.loads(source.read_text())
    root, out = Path(p['dataset_root']), Path(p['output'])
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / 'meta/info.json').read_text())
    raw_tasks = {r['task_index']: r['task'] for r in
                 map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
    split = split_episodes(manifest, p['split']['seed'], p['split']['validation_fraction'])
    split.update(source_manifest_sha256=sha256(source), source_revision=manifest['source_revision'])
    assert len(split['train']) == p['split']['expected_train_episodes']
    assert len(split['validation']) == p['split']['expected_validation_episodes']
    train_ids = {e['episode_id'] for e in split['train']}
    val_ids = {e['episode_id'] for e in split['validation']}
    assert not train_ids & val_ids
    split_path = Path('configs/mini_vla_split_20261015_v1.json')
    write_once(split_path, split)
    arrays = {'state': [], 'actions': []}
    rows, frames = [], Counter()
    for partition in ['train', 'validation']:
        for e in split[partition]:
            path = episode_path(root, info, e['episode_id'])
            table = pq.read_table(path, columns=['state', 'actions', 'episode_index', 'frame_index', 'task_index', 'timestamp'],
                                  use_threads=False)
            n = e['length']
            assert table.num_rows == n
            assert table['episode_index'].to_pylist() == [e['episode_id']] * n
            assert table['frame_index'].to_pylist() == list(range(n))
            raw_ids = set(table['task_index'].to_pylist())
            assert len(raw_ids) == 1 and raw_tasks[next(iter(raw_ids))] == e['instruction']
            np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(n) / info['fps'], atol=2e-6, rtol=1e-6)
            numeric = {key: np.asarray(table[key].to_pylist(), dtype=np.float32) for key in arrays}
            for key, dim in [('state', 8), ('actions', 7)]:
                assert numeric[key].shape == (n, dim) and np.isfinite(numeric[key]).all()
                if partition == 'train':
                    arrays[key].append(numeric[key])
            frames[partition] += n
            rows.append({'partition': partition, 'episode_id': e['episode_id'], 'frames': n,
                         'spatial_task_id': e['task_id'], 'source_task_id': next(iter(raw_ids)),
                         'file': str(path), 'bytes': path.stat().st_size,
                         'action_min': float(numeric['actions'].min()), 'action_max': float(numeric['actions'].max())})
    assert sum(frames.values()) == manifest['frames']
    stats = fit_normalization({k: np.concatenate(v) for k, v in arrays.items()})
    vocabulary = build_vocabulary(split['train'])
    preprocessing = {'normalization': stats, 'vocabulary': vocabulary,
                     'image_size': p['preprocessing']['image_size'],
                     'language_length': p['preprocessing']['max_language_length'],
                     'fit_episode_ids': sorted(train_ids), 'fit_frames': frames['train'],
                     'split_sha256': sha256(split_path), 'protocol_sha256': sha256(protocol_path)}
    write_once(out / 'preprocessing.json', preprocessing)
    write_once(out / 'numeric_audit.json', {'status': 'PASS', 'episodes': rows, 'frames': dict(frames),
                                         'total_episodes': len(rows), 'total_frames': sum(frames.values()),
                                         'all_numeric_finite': True, 'id_order_timestamp_checks': 'PASS'})
    dataset = DemonstrationDataset(root, split['train'], vocabulary, stats)
    # A declared teaching example, not a performance-evaluation sample.
    candidates = [e for e in split['train'] if e['task_id'] == 0]
    example = candidates[len(candidates) // 2]
    offset = sum(e['length'] for e in split['train'] if e['episode_id'] < example['episode_id'])
    frame = example['length'] // 2
    sample = dataset[offset + frame]
    path = episode_path(root, info, example['episode_id'])
    raw = dataset._table(example).slice(frame, 1).to_pylist()[0]
    for key in ['image', 'wrist_image']:
        target = out / f'example_{key}.png'
        blob = raw[key]['bytes']
        if target.exists():
            assert target.read_bytes() == blob
        else:
            target.write_bytes(blob)
    example_record = {
        **example, 'frame_index': frame, 'source_file': str(path), 'source_file_sha256': sha256(path),
        'raw_state': raw['state'], 'raw_action': raw['actions'],
        'normalized_state': sample['observations']['state'].tolist(),
        'normalized_action': sample['target_actions'].tolist(),
        'token_ids': sample['observations']['token_ids'].tolist(),
        'language_mask': sample['observations']['language_mask'].tolist(),
        'observation_shapes': {k: list(v.shape) for k, v in sample['observations'].items()},
        'target_shape': list(sample['target_actions'].shape),
        'image_orientation': 'embedded images kept as stored; online simulator adapter must orient raw camera frames before calling this policy',
    }
    write_once(out / 'example.json', example_record)
    print(json.dumps({'status': 'PASS', 'train_episodes': len(train_ids), 'validation_episodes': len(val_ids),
                      'frames': dict(frames), 'vocabulary_size': len(vocabulary),
                      'example_episode': example['episode_id'], 'example_frame': frame,
                      'elapsed_seconds': round(time.monotonic() - start, 3)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
