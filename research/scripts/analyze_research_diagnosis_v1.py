#!/usr/bin/env python3
"""Complete paired development statistics; no confirmation access or outcome-driven selection."""
import argparse
import collections
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import numpy as np

ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])


def write(p, d):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def preflight(out, plan):
    from PIL import Image
    old_dir = ROOT / plan['old_eval']
    old = {r['pair_id']: r for r in json.loads((old_dir / 'result.json').read_text())['episodes']}
    results = [json.loads((out / 'preflight' / ('episodes_' + mode) / 'result.json').read_text()) for mode in ('off', 'full')]
    assert all(r['status'] == 'PASS' and len(r['episodes']) == 2 for r in results)
    records = []
    for case in plan['preflight_cases']:
        key = case['pair_id']
        rows = [next(r for r in result['episodes'] if r['pair_id'] == key) for result in results]
        d0, d1 = [pathlib.Path(r['episode_dir']) for r in rows]
        with np.load(d0 / 'policy_trace.npz') as off, np.load(d1 / 'policy_trace.npz') as full:
            differences = {}
            for name in ('action', 'robot_state', 'action_chunk'):
                assert off[name].shape == full[name].shape, (key, name, 'shape')
                np.testing.assert_allclose(off[name], full[name], atol=1e-6, rtol=0, err_msg=f'{key}/{name}')
                differences[name] = float(np.max(np.abs(off[name] - full[name])))
            assert np.array_equal(off['image_sha256'], full['image_sha256']), key
            assert np.array_equal(off['done'], full['done'])
        assert rows[0]['steps'] == rows[1]['steps'] == old[key]['steps']
        assert rows[0]['success_at_220'] == rows[1]['success_at_220'] == old[key]['task_success']
        original = np.asarray(Image.open(old_dir / (key + '_first.png')).convert('RGB'))
        assert hashlib.sha256(original.tobytes()).hexdigest() == rows[1]['initial_image_sha256']
        assert np.array_equal(original, np.asarray(Image.open(d1 / 'first.png').convert('RGB')))
        with np.load(d1 / 'oracle.npz') as oracle:
            assert bool(oracle['goal_predicate'][-1].all()) == rows[1]['task_success']
            assert len(oracle['qpos']) == rows[1]['steps'] + 1
            assert np.max(np.ptp(oracle['object_position'], axis=0)) > 0, 'Telemetry must have nonconstant positions'
        records.append({'pair_id': key, 'success': rows[1]['task_success'], 'steps': rows[1]['steps'],
                        'identical_all_observation_hashes': True, 'max_abs_differences': differences,
                        'matches_historical_result_and_first_image': True})
    assert {r['success'] for r in records} == {False, True}, 'Require positive and negative controls'
    audit = {'status': 'PASS', 'created_at': datetime.datetime.now().astimezone().isoformat(), 'episodes': records,
             'checks': 'Original image and outcome; telemetry off/on full trajectory equivalence; finite and time-varying labels; original success predicate; read-only simulator and NumPy RNG assertions inside capture.'}
    write(out / 'preflight_acceptance.json', audit)
    return audit


def proxies(episode, threshold):
    ep = pathlib.Path(episode['episode_dir'])
    meta = json.loads((ep / 'oracle_metadata.json').read_text())
    with np.load(ep / 'oracle.npz') as raw:
        # Primary diagnosis always stops at 220, even if baseline has a 440-step tail.
        n = min(episode['steps'], 220) + 1
        pos = raw['object_position'][:n]
        contact = raw['bilateral_finger_contact'][:n]
        target = meta['object_names'].index(meta['target_object'])
        destination = meta['object_names'].index(meta['destination_object'])
        bowl_others = [i for i, name in enumerate(meta['object_names']) if 'bowl' in name and i != target]
        target_pos = pos[:, target]
        xy = np.linalg.norm(target_pos[:, :2] - pos[:, destination, :2], axis=1)
        lift = target_pos[:, 2] - target_pos[0, 2]
        displacement = np.linalg.norm(target_pos - target_pos[0], axis=1)
        stall = False
        window = threshold['stall_window_steps']
        # Must first encounter target contact; avoid classifying the initial approach as stall.
        for end in range(window, n):
            segment = target_pos[end-window:end+1]
            if contact[:end+1, target].any() and np.linalg.norm(segment[-1] - segment[0]) < threshold['stall_target_net_motion_m'] and np.linalg.norm(segment - segment[0], axis=1).max() < threshold['stall_target_max_excursion_m']:
                stall = True
                break
        target_contact = contact[:, target]
        return {'target_bilateral_contact_ever': bool(target_contact.any()),
                'other_bowl_bilateral_contact_ever': bool(contact[:, bowl_others].any()) if bowl_others else False,
                'target_lift_4cm_ever': bool((lift >= threshold['target_lift_m']).any()),
                'target_displacement_5cm_ever': bool((displacement >= threshold['transport_displacement_m']).any()),
                'target_near_destination_xy_5cm_ever': bool((xy <= threshold['near_destination_xy_m']).any()),
                'target_contact_absent_after_contact': bool(target_contact.any() and not target_contact[-1]),
                'target_motion_stall_after_contact': stall,
                'minimum_target_destination_xy_m': float(xy.min()), 'maximum_target_lift_m': float(lift.max()),
                'labels_are_nonexclusive_descriptive_proxies': True}


def analyze(out, plan):
    split = json.loads((ROOT / plan['split_manifest']).read_text())
    cases = split['development']
    expected = {c['pair_id'] for c in cases}
    all_results = {}
    video_count = 0
    for job in plan['jobs'][1:]:
        result = json.loads((out / job['name'] / 'episodes/result.json').read_text())
        assert result['status'] == 'PASS' and result['expected_episodes'] == 200
        rows = {r['pair_id']: r for r in result['episodes']}
        assert len(result['episodes']) == len(rows) == 200 and set(rows) == expected
        assert result['actual_egl_render_uuid'] == result['render_mapping']['selected_gpu']['uuid']
        for row in rows.values():
            ep = pathlib.Path(row['episode_dir'])
            n = row['steps']
            assert row['status'] in ('TASK_SUCCESS', 'STEP_LIMIT_REACHED')
            with np.load(ep / 'policy_trace.npz') as tr, np.load(ep / 'oracle.npz') as oracle:
                assert tr['action'].shape == (n, 7) and tr['robot_state'].shape == (n+1, 8)
                assert tr['action_chunk'].shape == (row['policy_calls'], 10, 7)
                assert np.array_equal(tr['call_step'], np.arange(0, n, job['replan_steps']))
                for name in ('action', 'robot_state', 'action_chunk', 'inference_seconds'):
                    assert np.isfinite(tr[name]).all()
                assert oracle['goal_predicate'].shape[0] == n+1
                assert np.array_equal(oracle['goal_predicate'][1:].all(axis=1), tr['done'])
                assert bool(tr['done'][:220].any()) == row['success_at_220']
                assert n <= job['max_steps']
            for camera in ('external', 'wrist'):
                probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                        'stream=width,height,nb_frames', '-of', 'json', str(ep / (camera + '.mp4'))],
                                       capture_output=True, text=True, check=True, timeout=30)
                stream = json.loads(probe.stdout)['streams'][0]
                assert (stream['width'], stream['height'], int(stream['nb_frames'])) == (224, 224, n+1)
                video_count += 1
        all_results[job['name']] = rows
    baseline = all_results['ckpt30k_prefix5']
    for rows in all_results.values():
        for key in expected:
            assert rows[key]['initial_image_sha256'] == baseline[key]['initial_image_sha256'], key
    summaries = {}
    for name, rows in all_results.items():
        ordered = [rows[c['pair_id']] for c in cases]
        successes = sum(r['success_at_220'] for r in ordered)
        # Count calls only within the primary budget, including in extended episodes.
        prefix = next(j['replan_steps'] for j in plan['jobs'] if j['name'] == name)
        calls = sum((min(r['steps'], 220) + prefix - 1) // prefix for r in ordered)
        primary_infer = 0.0
        for r in ordered:
            with np.load(pathlib.Path(r['episode_dir']) / 'policy_trace.npz') as tr:
                primary_infer += float(tr['inference_seconds'][tr['call_step'] < 220].sum())
        summaries[name] = {'n_used': 200, 'n_expected': 200, 'successes_at_220': successes,
                           'success_rate_at_220': successes/200,
                           'per_task_successes': [sum(r['success_at_220'] for r in ordered if r['task_id'] == t) for t in range(10)],
                           'primary_policy_calls': calls, 'primary_inference_seconds': primary_infer,
                           'calls_per_primary_success': calls/successes if successes else None,
                           'total_episode_wall_seconds_including_extensions': sum(r['elapsed_seconds'] for r in ordered),
                           'late_successes_221_to_440': sum(r['first_success_step'] is not None and r['first_success_step'] > 220 for r in ordered)}
    rng = np.random.default_rng(plan['primary_analysis']['bootstrap_seed'])
    draws = rng.integers(0, 10, size=(plan['primary_analysis']['bootstrap_draws'], 10))
    comparisons = {}
    pairs = [('30k-20k', 'ckpt30k_prefix5', 'ckpt20k_prefix5'), ('20k-10k', 'ckpt20k_prefix5', 'ckpt10k_prefix5'),
             ('prefix1-prefix5', 'ckpt30k_prefix1', 'ckpt30k_prefix5'), ('prefix10-prefix5', 'ckpt30k_prefix10', 'ckpt30k_prefix5')]
    for label, variant, reference in pairs:
        a, b = all_results[variant], all_results[reference]
        diffs = [np.mean([int(a[c['pair_id']]['success_at_220']) - int(b[c['pair_id']]['success_at_220']) for c in cases if c['task_id'] == t]) for t in range(10)]
        interval = np.quantile(np.asarray(diffs)[draws].mean(axis=1), [.025, .975])
        comparisons[label] = {'paired_episodes': 200, 'task_clusters': 10,
                              'success_difference': float(np.mean(diffs)), 'task_cluster_bootstrap_95pct': interval.tolist(),
                              'wins': sum(a[k]['success_at_220'] and not b[k]['success_at_220'] for k in expected),
                              'losses': sum(b[k]['success_at_220'] and not a[k]['success_at_220'] for k in expected),
                              'exploratory_unadjusted_interval': True}
    diagnoses = []
    for case in cases:
        row = baseline[case['pair_id']]
        diagnoses.append({'pair_id': case['pair_id'], 'task_id': case['task_id'], 'success_at_220': row['success_at_220'],
                          'late_success': row['first_success_step'] is not None and row['first_success_step'] > 220,
                          **proxies(row, plan['diagnostic_thresholds'])})
    group_counts = {}
    for label, success in [('primary_success', True), ('primary_failure', False)]:
        subset = [r for r in diagnoses if r['success_at_220'] == success]
        fields = ['target_bilateral_contact_ever', 'other_bowl_bilateral_contact_ever', 'target_lift_4cm_ever',
                  'target_displacement_5cm_ever', 'target_near_destination_xy_5cm_ever', 'target_motion_stall_after_contact', 'late_success']
        group_counts[label] = {'episodes': len(subset), 'nonexclusive_proxy_counts': {f: sum(r[f] for r in subset) for f in fields}}
    old = json.loads((ROOT / plan['old_eval'] / 'result.json').read_text())['episodes']
    old_checks = [{'pair_id': r['pair_id'], 'matches_success': baseline[r['pair_id']]['success_at_220'] == r['task_success'],
                   'matches_first_image': baseline[r['pair_id']]['initial_image_sha256'] == hashlib.sha256(__import__('PIL.Image', fromlist=['Image']).open(ROOT / plan['old_eval'] / (r['pair_id'] + '_first.png')).convert('RGB').tobytes()).hexdigest()}
                  for r in old]
    analysis = {'status': 'COMPLETE', 'generated_at': datetime.datetime.now().astimezone().isoformat(),
                'n_policy_episodes_used': 1000, 'n_policy_episodes_expected': 1000, 'unique_development_scenarios': 200,
                'confirmation_episodes_used': 0, 'training_seed_count': 1, 'summaries': summaries,
                'paired_comparisons': comparisons, 'baseline_failure_proxies': group_counts,
                'old_anchor_rechecks': old_checks, 'video_container_metadata_checked': video_count,
                'video_check_scope': 'Frame count/dimensions in container metadata; preflight observation hashes are computed before video compression.',
                'scientific_scope': 'Exploratory development diagnosis only. Finger contacts and thresholds are proxies. Multiple frames/variants are not independent scenarios; no causal recovery claim.',
                'next_stage': 'Review failure mechanisms and strong baseline choice; freeze branch reproduction tests and recovery candidates before new data. Do not open confirmation split.'}
    write(out / 'diagnostic_episode_proxies.json', {'n_used': 200, 'n_expected': 200, 'episodes': diagnoses})
    write(out / 'ANALYSIS.json', analysis)
    lines = ['# VLA 开发集首轮诊断', '', '自动统计已完成，研究结论待原会话核验。', '',
             '样本：1000/1000 次策略评测，200 个共同开发场景、10 个任务；确认集使用 0/300。原 VLA 训练种子只有 1 个。', '',
             '| 条件 | 220 步成功数 / 200 | 成功率 | 220 步内策略调用 | 221–440 步晚成功 |',
             '|---|---:|---:|---:|---:|']
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['successes_at_220']}/200 | {s['success_rate_at_220']:.1%} | {s['primary_policy_calls']} | {s['late_successes_221_to_440']} |")
    lines += ['', '晚成功仅用于时间截断诊断，不计为同预算方法收益。前缀更短会增加推理成本。', '',
              '| 配对比较 | 成功率差 | 按任务聚类 95% 区间 | 净胜 / 净负场景 |', '|---|---:|---:|---:|']
    for name, s in comparisons.items():
        lo, hi = s['task_cluster_bootstrap_95pct']
        lines.append(f"| {name} | {s['success_difference']*100:.1f} pp | [{lo*100:.1f}, {hi*100:.1f}] pp | {s['wins']} / {s['losses']} |")
    lines += ['', '区间是未校正的探索性开发结果，不含训练种子不确定性。几何和双指接触标签不等同于稳定抓取或已证实的失败根因。', '',
              '逐案例标签见 `diagnostic_episode_proxies.json`；完整校验、旧样本复核及计费口径见 `ANALYSIS.json`。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return analysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    out = pathlib.Path(a.output)
    plan = json.loads(pathlib.Path(a.plan).read_text())
    preflight(out, plan)
    result = {'status': 'PASS'} if a.preflight_only else analyze(out, plan)
    print(json.dumps({'status': result['status'], 'preflight_only': a.preflight_only}), flush=True)


if __name__ == '__main__':
    main()
