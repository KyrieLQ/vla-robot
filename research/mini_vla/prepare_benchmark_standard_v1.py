"""Prepare four-suite metadata only; no simulator, torch, network or GPU calls."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_once(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen metadata changed: {path}'
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')


def main():
    data = ROOT / 'datasets/lerobot/physical-intelligence/libero'
    lib = ROOT / 'repos/openpi/third_party/libero/libero/libero'
    map_path = lib / 'benchmark/libero_suite_task_map.py'
    task_map = ast.literal_eval(ast.parse(map_path.read_text()).body[0].value)
    episodes = [json.loads(s) for s in (data / 'meta/episodes.jsonl').read_text().splitlines()]
    old_cases = json.loads((ROOT / 'configs/mini_vla_spatial_cases_20261019_v1.json').read_text())['cases']
    spatial_partition = {(c['task_id'], c['initial_state_index']): c['partition'] for c in old_cases}
    split = {'seed': 20261015, 'validation_fraction': 0.1, 'train': [], 'validation': []}
    inventory, cases, assets, seen = [], [], {}, set()
    for suite in SUITES:
        summary = dict(suite=suite, tasks=10, episodes=0, frames=0, local_files=0, local_bytes=0, missing_paths=[])
        for task_id, name in enumerate(task_map[suite]):
            language = re.sub(r'^[A-Z_]+SCENE\d+_', '', name).replace('_', ' ')
            rows = [e for e in episodes if e['tasks'] == [language]]
            assert len(rows) >= 2
            rows.sort(key=lambda e: hashlib.sha256(f"20261015:{e['episode_index']}".encode()).hexdigest())
            for i, e in enumerate(rows):
                eid = e['episode_index']
                assert eid not in seen
                seen.add(eid)
                item = dict(episode_id=eid, length=e['length'], task_id=task_id, instruction=language, suite=suite)
                split['validation' if i < math.ceil(len(rows) * 0.1) else 'train'].append(item)
                path = data / f'data/chunk-{eid // 1000:03d}/episode_{eid:06d}.parquet'
                summary['episodes'] += 1
                summary['frames'] += e['length']
                if path.is_file():
                    summary['local_files'] += 1
                    summary['local_bytes'] += path.stat().st_size
                else:
                    summary['missing_paths'].append(str(path.relative_to(data)))
            for folder, suffix in [('bddl_files', '.bddl'), ('init_files', '.pruned_init')]:
                path = lib / folder / suite / (name + suffix)
                assets[str(path.relative_to(ROOT))] = digest(path)
            # Existing Spatial partition is immutable. New suites use fixed, stratified IDs.
            dev = set()
            for start in range(0, 50, 10):
                ordered = sorted(range(start, start + 10), key=lambda j: hashlib.sha256(f'20261019:{suite}:{task_id}:{j}'.encode()).hexdigest())
                dev.update(ordered[:4])
            for idx in range(50):
                partition = spatial_partition[(task_id, idx)] if suite == 'libero_spatial' else ('development' if idx in dev else 'confirmation')
                cases.append(dict(suite=suite, task_id=task_id, task_name=name, initial_state_index=idx,
                                  partition=partition, final_environment_seed=7))
        inventory.append(summary)
    assert len(seen) == len(episodes) == 1693
    assert len(cases) == 2000 and len(assets) == 80
    for part in ['train', 'validation']:
        split[part].sort(key=lambda x: x['episode_id'])
        old = json.loads((ROOT / 'configs/mini_vla_split_20261015_v1.json').read_text())[part]
        current = [{k: v for k, v in row.items() if k != 'suite'} for row in split[part] if row['suite'] == 'libero_spatial']
        assert current == old, 'Spatial demonstration split changed'
    for suite in SUITES:
        for tid in range(10):
            assert sum(c['partition'] == 'development' for c in cases if c['suite'] == suite and c['task_id'] == tid) == 20
    source = dict(repo='physical-intelligence/libero', revision='a4336d589d589045d1c56423ffdf3b88a0e19b1f',
                  metadata_sha256={p.name: digest(p) for p in [data/'meta/info.json', data/'meta/episodes.jsonl', data/'meta/tasks.jsonl']})
    split['source'] = source
    save_once(ROOT/'configs/mini_vla_four_suite_demo_split_20261019_v1.json', split)
    save_once(ROOT/'configs/mini_vla_four_suite_cases_20261019_v1.json',
              dict(task_order_index=0, cases=cases, asset_sha256=assets,
                   access='Metadata only. Confirmation execution requires frozen method/checkpoints/settings; no release granted.'))
    audit = dict(source=source, suites=inventory, metadata_episodes=len(seen),
                 local_files=sum(x['local_files'] for x in inventory),
                 missing_files=sum(len(x['missing_paths']) for x in inventory),
                 train_episodes=len(split['train']), validation_episodes=len(split['validation']),
                 full_uncompressed_image_cache_gib=sum(x['frames'] for x in inventory)*2*3*128*128/2**30,
                 audit_scope='Metadata mapping, exact-path file presence/size and 80 asset hashes; no parquet-content integrity audit or initial-state deserialization',
                 old_spatial_demo_and_eval_partitions_unchanged=True)
    save_once(ROOT/'reports/mini_vla_four_suite_inventory_20261019.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ['source', 'suites']}, indent=2))


if __name__ == '__main__':
    main()
