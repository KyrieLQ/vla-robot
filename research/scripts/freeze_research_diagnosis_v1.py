#!/usr/bin/env python3
"""Create once, before rollouts. Hash-stratified indices, with all previously seen cases in development."""
import datetime
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    seed = 20261014
    old = [2, 12, 22, 32, 42]
    dev, confirm = [], []
    def case(t, i):
        return {'pair_id': f'task{t}_init{i}', 'task_id': t, 'initial_state_index': i,
                'env_seed': 1000 + 100*t + i, 'policy_seed': 2000 + 100*t + i}
    for t in range(10):
        chosen = set(old)
        for block in range(5):
            remaining = [i for i in range(10*block, 10*(block+1)) if i not in chosen]
            ranked = sorted(remaining, key=lambda i: hashlib.sha256(f'{seed}:{t}:{i}'.encode()).hexdigest())
            chosen.update(ranked[:3])
        assert len(chosen) == 20
        dev.extend(case(t, i) for i in sorted(chosen))
        confirm.extend(case(t, i) for i in range(50) if i not in chosen)
    assert len(dev) == 200 and len(confirm) == 300
    stamp = datetime.datetime.now().astimezone().isoformat()
    split_path = ROOT / 'configs/research_split_20261014_v1.json'
    dump(split_path, {'created_at': stamp, 'seed': seed, 'sampling': 'Per task, five index strata of ten; retain old index 10*b+2 and select three smallest SHA256(seed:task:index) in each stratum.',
                     'already_inspected_indices_per_task': old, 'development': dev, 'confirmation': confirm,
                     'confirmation_policy': 'No rollouts, videos or outcomes until method and thresholds are frozen. This is a development holdout, not a claim of disjointness from training demonstration states.'})
    jobs = [
        {'name': 'preflight', 'checkpoint_step': 30000, 'replan_steps': 5, 'max_steps': 220, 'cases': 'preflight'},
        {'name': 'ckpt30k_prefix5', 'checkpoint_step': 30000, 'replan_steps': 5, 'max_steps': 440, 'cases': 'development'},
        {'name': 'ckpt10k_prefix5', 'checkpoint_step': 10000, 'replan_steps': 5, 'max_steps': 220, 'cases': 'development'},
        {'name': 'ckpt20k_prefix5', 'checkpoint_step': 20000, 'replan_steps': 5, 'max_steps': 220, 'cases': 'development'},
        {'name': 'ckpt30k_prefix1', 'checkpoint_step': 30000, 'replan_steps': 1, 'max_steps': 220, 'cases': 'development'},
        {'name': 'ckpt30k_prefix10', 'checkpoint_step': 30000, 'replan_steps': 10, 'max_steps': 220, 'cases': 'development'},
    ]
    plan = {'version': 1, 'created_at': stamp, 'authorization_text': '那开始吧，正好把空闲的gpu利用起来',
            'originating_thread': 'PUBLIC_REPRODUCTION',
            'objective': 'Bounded failure diagnosis for task-progress-aware selective recovery; first stage of the approved 1–2 month research plan.',
            'split_manifest': str(split_path.relative_to(ROOT)), 'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'jobs': jobs, 'preflight_cases': [case(0, 2), case(4, 22)],
            'preflight_protocol': 'Run both cases with telemetry off and on; require identical image hashes, success, steps; actions/state/chunks allclose atol=1e-6 rtol=0; compare original first image and 220-step result. Any mismatch blocks the study and requires diagnosis, no outcome-driven retries.',
            'primary_max_steps': 220, 'wait_steps': 10, 'denoising_steps': 10, 'action_horizon': 10,
            'checkpoint_root': 'checkpoints/pi05_libero_lora_3090x2_train/formal_spatial_20261013_v1',
            'old_eval': 'results/formal_pipeline_20261013_v2/trained_eval/episodes',
            'allowed_physical_gpus': [0, 1, 2, 3], 'maximum_simultaneous_gpus': 2,
            'initial_worker_count': 1, 'admission': 'Two samples plus immediate recheck, no compute PIDs, <=512MiB used, >=95% free, <=5% utilization. One finite admission per job; no persistent waiting.',
            'project_budget_gib': 120, 'filesystem_reserve_gib': 100, 'new_artifact_budget_gib': 8,
            'queue_timeout_seconds': 36000, 'job_timeout_seconds': 10800,
            'expected_policy_episodes': 1000, 'extra_preflight_episodes': 4,
            'persistent_idle_monitor': False, 'periodic_chat_notifications': False,
            'new_gradient_updates_this_queue': 0, 'confirmation_episodes_this_queue': 0,
            'primary_analysis': {'unit': 'episode paired by task and initial state; 10 task clusters',
                                 'bootstrap_seed': 20261014, 'bootstrap_draws': 10000,
                                 'contrasts': ['30k-20k', '20k-10k', 'prefix1-prefix5', 'prefix10-prefix5'],
                                 'intervals': 'Exploratory task-cluster percentile 95% intervals, not adjusted confirmatory claims; disclose one VLA training seed.',
                                 'costs': ['policy calls', 'inference seconds', 'episode wall seconds', 'steps to first success'],
                                 'late_success': '220-to-440 extension for baseline only, reported separately; never treated as a 220-step method gain'},
            'diagnostic_thresholds': {'target_lift_m': 0.04, 'transport_displacement_m': 0.05,
                                      'near_destination_xy_m': 0.05, 'stall_window_steps': 40,
                                      'stall_target_net_motion_m': 0.01, 'stall_target_max_excursion_m': 0.02,
                                      'description': 'Descriptive non-exclusive geometric/contact proxies, not causal labels; success predicates unchanged. Fixed before new outcomes.'},
            'interpretation_rules': ['No method superiority claim from development results or interim totals.',
                                     'Large delay-only fraction strengthens the timeout explanation; do not market extended budget as recovery.',
                                     'Better short prefix must be reported with its additional inference cost.',
                                     'Checkpoint improvements suggest under-training as a competitor, not proof that recovery is unnecessary.',
                                     'Contact/position associations do not prove intervention benefit; next stage must validate complete-state branch reproduction and paired intervention outcomes.',
                                     'No new full VLA training or confirmation evaluation is part of this queue. Freeze next-stage cases, actions, thresholds and remaining budget before data collection.'],
            'telemetry': ['two compressed cameras', 'actions and full predicted chunks', 'robot state', 'object/fixture poses', 'bilateral finger contact proxy', 'object contacts', 'original goal predicate', 'policy call cost'],
            'information_boundary': 'Oracle metadata and arrays remain evaluator-side; inference payload contains only images, language, robot state and the pre-existing episode RNG seed.',
            'auto_followup': 'Verify complete counts and preflight, compute paired cluster statistics and failure proxies, write ANALYSIS.json and REPORT.md; one completion/failure event to originating conversation for review and necessary authorized next stage.'}
    path = ROOT / 'configs/research_diagnosis_20261014_v1.json'
    dump(path, plan)
    print(json.dumps({'plan': str(path), 'development': len(dev), 'confirmation': len(confirm), 'policy_episodes': 1000, 'new_gradient_updates': 0}))


if __name__ == '__main__':
    main()
