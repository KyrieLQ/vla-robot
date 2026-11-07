"""Bounded CPU acceptance for data and the untrained M1 model, not a benchmark."""
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import signal
import time
import unittest

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from .data import (DemonstrationDataset, denormalize, episode_path, normalize, sha256)
from .model import MiniVLA, ModelConfig, masked_mse
from .policy import MiniVLAPolicy


def main():
    started = time.monotonic()
    protocol_path = Path('configs/mini_vla_m0_20261015_v1.json')
    protocol = json.loads(protocol_path.read_text())
    out = Path(protocol['output'])
    if (out / 'acceptance.json').exists():
        raise RuntimeError('Acceptance already recorded; use a new record for another attempt')
    def timeout(signum, frame):
        raise TimeoutError('CPU verification time cap exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(protocol['verification']['max_seconds_per_command'])
    torch.set_num_threads(protocol['verification']['threads'])
    torch.manual_seed(protocol['model']['seed'])
    split = json.loads(Path('configs/mini_vla_split_20261015_v1.json').read_text())
    pre = json.loads((out / 'preprocessing.json').read_text())
    assert pre['protocol_sha256'] == sha256(protocol_path)
    assert pre['split_sha256'] == sha256(Path('configs/mini_vla_split_20261015_v1.json'))
    assert pre['fit_episode_ids'] == sorted(e['episode_id'] for e in split['train'])
    dataset_root = Path(protocol['dataset_root'])
    dataset = DemonstrationDataset(dataset_root, split['train'], pre['vocabulary'], pre['normalization'])

    # A second accumulation path verifies stored train-only means and scales.
    count = 0
    sums = {'state': np.zeros(8), 'actions': np.zeros(7)}
    squares = {k: np.zeros_like(v) for k, v in sums.items()}
    for e in split['train']:
        table = pq.read_table(episode_path(dataset_root, dataset.info, e['episode_id']),
                              columns=['state', 'actions'], use_threads=False)
        count += table.num_rows
        for key in sums:
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)
            sums[key] += values.sum(axis=0)
            squares[key] += np.square(values).sum(axis=0)
    assert count == pre['fit_frames']
    for key in sums:
        mean = sums[key] / count
        scale = np.sqrt(np.maximum(0, squares[key] / count - mean ** 2))
        scale[scale < 1e-6] = 1
        np.testing.assert_allclose(mean, pre['normalization'][key]['mean'], atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(scale, pre['normalization'][key]['scale'], atol=1e-10, rtol=1e-8)

    example = json.loads((out / 'example.json').read_text())
    offset = sum(e['length'] for e in split['train'] if e['episode_id'] < example['episode_id'])
    sample = dataset[offset + example['frame_index']]
    observations = {k: v.unsqueeze(0) for k, v in sample['observations'].items()}
    model = MiniVLA(ModelConfig(len(pre['vocabulary'])))
    prediction = model(**observations)
    assert prediction.shape == (1, 1, 7) and torch.isfinite(prediction).all()
    loss = masked_mse(prediction, sample['target_actions'].unsqueeze(0), sample['action_mask'].unsqueeze(0))
    loss.backward()
    gradients = {}
    for name, module in [('vision', model.vision), ('language', model.language),
                         ('proprioception', model.proprioception), ('fusion', model.fusion), ('action_head', model.action_head)]:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        gradients[name] = sum(float(g.abs().sum()) for g in grads)
        assert gradients[name] > 0

    # Input sensitivity at initialization proves connectivity, not semantic skill.
    delta = {}
    with torch.no_grad():
        for key in ['image', 'wrist_image', 'state', 'token_ids']:
            changed = {k: v.clone() for k, v in observations.items()}
            if key == 'token_ids':
                changed[key][observations['language_mask']] = 1
            elif key == 'state':
                changed[key] = changed[key] + 1
            else:
                changed[key] = 1 - changed[key]
            delta[key] = float((model(**changed) - prediction).abs().max())
            assert delta[key] > 1e-8
        padded = {k: v.clone() for k, v in observations.items()}
        padded['token_ids'][~padded['language_mask']] = 1
        torch.testing.assert_close(model(**padded), prediction, atol=0, rtol=0)
    try:
        model.predict(**observations, actions=sample['target_actions'])
    except TypeError:
        pass
    else:
        raise AssertionError('Inference unexpectedly accepts future action labels')

    checkpoint = out / 'untrained_model.pt'
    if checkpoint.exists():
        raise RuntimeError('Do not replace an existing model artifact')
    torch.save({'model_config': model.configuration(), 'state_dict': model.state_dict(),
                'status': 'UNTRAINED_ENGINEERING_ARTIFACT', 'preprocessing_sha256': sha256(out / 'preprocessing.json')}, checkpoint)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    clone = MiniVLA(ModelConfig(**restored['model_config']))
    clone.load_state_dict(restored['state_dict'])
    torch.testing.assert_close(clone(**observations), prediction, rtol=0, atol=0)

    # Raw deployment observations must produce the same normalized action as training preprocessing.
    obs = {'observation/image': np.array(Image.open(out / 'example_image.png').convert('RGB')),
           'observation/wrist_image': np.array(Image.open(out / 'example_wrist_image.png').convert('RGB')),
           'observation/state': np.array(example['raw_state'], dtype=np.float32),
           'prompt': example['instruction']}
    deployed = MiniVLAPolicy(clone, pre['vocabulary'], pre['normalization']).infer(obs)['actions']
    expected = denormalize(prediction.detach()[0], pre['normalization']['actions']).clamp(-1, 1).numpy()
    np.testing.assert_array_equal(deployed, expected)
    for key, raw in [('state', example['raw_state']), ('actions', example['raw_action'])]:
        stats = pre['normalization'][key]
        np.testing.assert_allclose(denormalize(normalize(raw, stats), stats).numpy(), raw, atol=2e-7, rtol=1e-6)

    suite = unittest.defaultTestLoader.discover('mini_vla/tests', pattern='test_*.py', top_level_dir='.')
    test_log = io.StringIO()
    unit = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
    (out / 'unit_checks.txt').write_text(test_log.getvalue())
    assert unit.wasSuccessful(), test_log.getvalue()

    # Positive learning control on synthetic inputs only; discard its fitted weights.
    control = protocol['verification']['synthetic_fit']
    torch.manual_seed(control['seed'])
    learner = MiniVLA(ModelConfig(len(pre['vocabulary'])))
    b = control['batch']
    synthetic = {k: v.repeat(b, *([1] * (v.ndim - 1))) for k, v in observations.items()}
    synthetic['image'] = torch.rand(b, 3, 128, 128)
    synthetic['wrist_image'] = torch.rand(b, 3, 128, 128)
    synthetic['state'] = torch.rand(b, 8)
    synthetic['token_ids'] = torch.full_like(synthetic['token_ids'], 2)
    synthetic['language_mask'] = torch.ones_like(synthetic['language_mask'])
    target = torch.tensor([0.4, -0.2, 0.1, 0.3, -0.4, 0.2, 0.8]).reshape(1, 1, 7).repeat(b, 1, 1)
    valid = torch.ones(b, 1, dtype=torch.bool)
    optimizer = torch.optim.Adam(learner.parameters(), lr=control['learning_rate'])
    curve = []
    for _ in range(control['steps']):
        optimizer.zero_grad(set_to_none=True)
        objective = masked_mse(learner(**synthetic), target, valid)
        curve.append(float(objective.detach()))
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(masked_mse(learner(**synthetic), target, valid))
    assert final < 0.2 * curve[0], (curve[0], final)

    result = {
        'status': 'PASS', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'device': 'cpu', 'threads': torch.get_num_threads(),
        'torch_version': torch.__version__, 'elapsed_seconds': time.monotonic() - started,
        'unit_checks_passed': unit.testsRun, 'split': {'train_episodes': len(split['train']), 'validation_episodes': len(split['validation']), 'train_frames': count},
        'train_only_statistics_independent_recalculation': 'PASS',
        'parameters': sum(p.numel() for p in model.parameters()),
        'parameters_by_module': {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()},
        'output_shape': list(prediction.shape), 'real_example_loss_untrained': float(loss.detach()),
        'gradient_l1_by_module': gradients, 'input_change_max_output_difference': delta,
        'input_sensitivity_scope': 'Connectivity only, not learned language or control competence',
        'label_exclusion': 'PASS', 'checkpoint_roundtrip': 'EXACT',
        'training_inference_preprocessing': 'EXACT', 'normalization_roundtrip': 'PASS',
        'synthetic_fit': {'steps': control['steps'], 'initial_mse': curve[0], 'final_mse': final, 'curve': curve, 'weights_retained': False},
        'real_data_optimizer_updates': 0, 'new_simulation_episodes': 0, 'confirmation_episodes': 0,
        'source_hashes': {str(p): sha256(p) for p in sorted(Path('mini_vla').rglob('*.py'))},
        'artifact_hashes': {str(p): sha256(p) for p in [protocol_path, Path('configs/mini_vla_split_20261015_v1.json'), out / 'preprocessing.json', checkpoint]},
    }
    used = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    result['artifact_bytes_before_report'] = used
    assert used + 65536 < protocol['verification']['new_artifact_cap_mib'] * 2 ** 20
    (out / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    signal.alarm(0)
    print(json.dumps({k: result[k] for k in ['status', 'parameters', 'unit_checks_passed', 'elapsed_seconds', 'real_data_optimizer_updates']}, ensure_ascii=False), flush=True)
    print(json.dumps({'synthetic_initial_mse': curve[0], 'synthetic_final_mse': final, 'input_sensitivity': delta}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
