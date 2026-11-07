"""Independent CPU-only final audit. Does not call the evaluator or simulator."""
import csv
import hashlib
import json
import re
import time
from collections import Counter, deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/mini_vla_final2000_20261021_run3'
OUT = RUN / 'review'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    plan = read(ROOT / 'configs/mini_vla_final2000_resume_20261021_v1.json')
    science = read(ROOT / plan['scientific_plan'])
    aggregate = read(RUN / 'analysis.json')
    assert read(RUN / 'status.json')['status'] == 'COMPLETE'
    assert sha(ROOT / plan['checkpoint']) == plan['checkpoint_sha256']
    # Frozen inputs include task descriptions/init files, adapter and checkpoint.
    for p, h in plan['input_sha256'].items():
        assert sha(ROOT / p) == h, p
    sources = read(RUN / 'source_manifest.json')
    for p, h in sources.items():
        assert sha(ROOT / p) == h, p
        assert sha(RUN / 'frozen_source' / p) == h, p
    selection = read(ROOT / science['selection'])
    for p, h in selection['source_acceptances'].items():
        assert sha(ROOT / p) == h, p
    cases = read(ROOT / science['case_manifest'])['cases']
    expected = {(c['suite'], c['task_id'], c['initial_state_index']): c for c in cases}
    assert len(cases) == len(expected) == 2000
    science_hash = sha(ROOT / plan['scientific_plan'])
    output_hashes = {}
    rows = []
    video_rows = []
    all_calls = 0
    all_steps = 0
    suite_reports = {}
    latency_samples = {}
    for worker, suites in enumerate(plan['worker_suites']):
        for suite in suites:
            directory = RUN / f'worker{worker}' / suite
            result = read(directory / 'result.json')
            assert result['expected'] == len(result['episodes']) == 500
            meta = result['metadata']
            assert meta['epoch'] == 3 and meta['cpu_threads'] == 1
            assert meta['checkpoint_sha256'] == plan['checkpoint_sha256']
            assert meta['adapter_sha256'] == sha(ROOT / 'mini_vla/policy.py')
            assert len(list(directory.glob('task*_init*'))) == 500
            log = (RUN / f'worker{worker}' / (suite + '.log')).read_text()
            events = re.findall(r'^' + suite + r' (\d+) (\d+) (completed|reused) (\d+)$', log, re.M)
            assert [(int(t), int(i), int(n)) for t, i, _, n in events] == [(t, i, t*50+i+1) for t in range(10) for i in range(50)]
            limit = next(x['max_action_steps'] for x in science['suites'] if x['name'] == suite)
            rpc = []
            group = []
            for ordinal, row in enumerate(result['episodes']):
                task, init = divmod(ordinal, 50)
                key = (suite, task, init)
                d = directory / f'task{task}_init{init}'
                assert row == read(d / 'episode.json')
                assert (row['suite'], row['task_id'], row['initial_state_index']) == key
                assert row['partition'] == expected[key]['partition']
                assert row['checkpoint_sha256'] == plan['checkpoint_sha256']
                assert row['scientific_plan_sha256'] == science_hash
                assert (row['action_horizon'], row['execute_steps'], row['environment_seed']) == (10, 5, 7)
                assert row['reset_protocol'] == 'upstream_sequential_per_task'
                identity = read(d / 'start.json')
                for k, v in identity.items():
                    assert row[k] == v, (key, k)
                for name in ['initial_qpos', 'initial_qvel']:
                    assert np.isfinite(row[name]).all()
                assert re.fullmatch('[0-9a-f]{64}', row['initial_image_sha256'])
                raw = (d / 'trajectory.jsonl').read_bytes()
                assert raw.endswith(b'\n')
                assert hashlib.sha256(raw).hexdigest() == row['trace_sha256']
                # Independent sequential state machine, no production validator import.
                queue = deque()
                step_count = call_count = 0
                done = False
                for line in raw.splitlines():
                    record = json.loads(line)
                    assert not done
                    if record['kind'] == 'call':
                        assert not queue and record['step'] == step_count and step_count % 5 == 0
                        chunk = np.asarray(record['actions'])
                        assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                        assert abs(chunk).max() <= 1.000001
                        assert re.fullmatch('[0-9a-f]{64}', record['input_sha256'])
                        assert np.isfinite([record['latency'], record['server_ms']]).all()
                        assert record['latency'] >= 0 and record['server_ms'] >= 0
                        if call_count:
                            rpc.append(record['latency'])
                        queue.extend(chunk[:5].tolist())
                        call_count += 1
                    else:
                        assert record['kind'] == 'step' and record['step'] == step_count and queue
                        assert record['action'] == queue.popleft()
                        assert len(record['action']) == 7 and len(record['state']) == 8
                        for k in ['state', 'qpos', 'qvel']:
                            assert np.isfinite(record[k]).all()
                        assert len(record['qpos']) == len(row['initial_qpos'])
                        assert len(record['qvel']) == len(row['initial_qvel'])
                        assert isinstance(record['done'], bool)
                        done = record['done']
                        step_count += 1
                        assert step_count <= limit
                assert step_count == row['steps'] and call_count == row['policy_calls']
                assert done == row['task_success'] and (done or step_count == limit)
                all_steps += step_count
                all_calls += call_count
                for fn in ['episode.json', 'start.json', 'trajectory.jsonl']:
                    output_hashes[str((d / fn).relative_to(RUN))] = sha(d / fn)
                assert aggregate['artifacts_sha256'][str(d / 'episode.json')] == sha(d / 'episode.json')
                if init in [0, 25]:
                    vp = d / 'external.mp4'
                    assert sha(vp) == row['video_sha256']
                    chosen = {0, step_count//2, step_count}
                    frames = []
                    n = 0
                    with imageio.get_reader(str(vp), ffmpeg_params=['-threads', '1']) as reader:
                        for idx, frame in enumerate(reader):
                            assert frame.shape == (256, 256, 3)
                            if idx in chosen:
                                frames.append((idx, Image.fromarray(frame)))
                            n += 1
                    assert n == step_count + 1, (key, n, step_count)
                    preview = OUT / f'{suite}_task{task}_init{init}.jpg'
                    strip = Image.new('RGB', (768, 286), 'white')
                    draw = ImageDraw.Draw(strip)
                    for col, (idx, im) in enumerate(frames):
                        strip.paste(im, (col*256, 30))
                        draw.text((col*256+5, 7), f'{suite} t{task}/i{init} step{idx} success={done}', fill='black')
                    strip.save(preview, quality=92)
                    video_rows.append(dict(suite=suite, task=task, init=init, success=done, frames=n, video=str(vp.relative_to(ROOT)), preview=str(preview.relative_to(ROOT)), sha256=sha(vp)))
                    output_hashes[str(vp.relative_to(RUN))] = sha(vp)
                else:
                    assert row['video_sha256'] is None and not (d / 'external.mp4').exists()
                compact = dict(suite=suite, task_id=task, initial_state_index=init, partition=row['partition'], success=int(done), steps=step_count, calls=call_count, language=row['language'])
                rows.append(compact)
                group.append(compact)
            parts = {}
            for part in ['development', 'confirmation']:
                subset = [r for r in group if r['partition'] == part]
                parts[part] = dict(successes=sum(r['success'] for r in subset), episodes=len(subset))
                assert parts[part] == aggregate['suites'][suite]['partitions'][part]
            tasks = [dict(task_id=t, successes=sum(r['success'] for r in group if r['task_id']==t), episodes=50,
                          language=next(r['language'] for r in group if r['task_id']==t)) for t in range(10)]
            successes = sum(r['success'] for r in group)
            assert successes == aggregate['suites'][suite]['successes']
            assert [{k:v for k,v in t.items() if k!='language'} for t in tasks] == aggregate['suites'][suite]['per_task']
            latency_samples[suite] = rpc
            med, p95 = np.quantile(rpc, [.5, .95])*1000
            assert abs(med-aggregate['suites'][suite]['rpc_median_ms']) < 1e-9
            assert abs(p95-aggregate['suites'][suite]['rpc_p95_ms']) < 1e-9
            suite_reports[suite] = dict(successes=successes, episodes=500, success_rate=successes/500, partitions=parts, per_task=tasks,
                                        total_steps=sum(r['steps'] for r in group), total_calls=sum(r['calls'] for r in group), rpc_median_ms=float(med), rpc_p95_ms=float(p95))
            print(suite, successes, '/500; independent trace/video audit passed', flush=True)
    assert len(rows) == 2000 and len(video_rows) == 80
    assert Counter(r['partition'] for r in rows) == {'development':800, 'confirmation':1200}
    # Byte preservation of all completed outputs and the single interrupted prefix.
    old = ROOT / plan['resume_source_run']
    reuse = read(RUN / 'resume_reuse_manifest.json')
    complete_reused = 0
    prefix_checks = []
    for f in reuse['files']:
        path = f['path']
        assert sha(old / path) == f['sha256']
        if f['mode'] == 'immutable_hardlink':
            assert sha(RUN / path) == f['sha256'], path
            if path.endswith('/episode.json'):
                complete_reused += 1
        else:
            old_bytes = (old/path).read_bytes()
            assert (RUN/path).read_bytes().startswith(old_bytes), path
            prefix_checks.append(path)
    assert complete_reused == 161
    part = 'worker1/libero_object/task1_init19/trajectory.jsonl'
    old_records = [json.loads(x) for x in (old/part).read_bytes().splitlines()]
    new_records = [json.loads(x) for x in (RUN/part).read_bytes().splitlines()]
    assert new_records[:len(old_records)] == old_records
    old_steps = [r for r in old_records if r['kind']=='step']
    old_calls = [r for r in old_records if r['kind']=='call']
    assert len(old_steps) == 51 and len(old_calls) == 11 and old_calls[-1]['step']==50
    next_records = new_records[len(old_records):len(old_records)+5]
    for offset, rec in enumerate(next_records[:4], 1):
        assert rec['kind']=='step' and rec['step']==50+offset
        assert rec['action']==old_calls[-1]['actions'][offset]
    assert next_records[4]['kind']=='call' and next_records[4]['step']==55
    rng = np.random.default_rng(20261021)
    suite_boots = []
    for suite in sorted(suite_reports):
        means = np.array([x['successes']/50 for x in suite_reports[suite]['per_task']])
        draws = rng.integers(0, 10, (10000, 10))
        sample = np.take(means, draws).mean(axis=1)
        suite_reports[suite]['task_cluster_descriptive_95_interval'] = np.quantile(sample, [.025, .975]).tolist()
        suite_boots.append(sample)
    ci = np.quantile(np.stack(suite_boots).mean(axis=0), [.025,.975]).tolist()
    np.testing.assert_allclose(ci, aggregate['task_cluster_descriptive_interval'], rtol=0, atol=1e-12)
    partitions = {p:dict(successes=sum(r['success'] for r in rows if r['partition']==p), episodes=sum(r['partition']==p for r in rows)) for p in ['development','confirmation']}
    # Partition intervals use their task clusters too; these are descriptive, not a new hypothesis test.
    partition_rng = np.random.default_rng(20261021)
    for part, report in partitions.items():
        boots = []
        for suite in sorted(suite_reports):
            means = []
            for task in range(10):
                subset = [r['success'] for r in rows if r['suite']==suite and r['task_id']==task and r['partition']==part]
                assert len(subset) == (20 if part=='development' else 30)
                means.append(np.mean(subset))
            boots.append(np.take(means, partition_rng.integers(0,10,(10000,10))).mean(1))
        report['task_cluster_descriptive_95_interval'] = np.quantile(np.mean(boots,axis=0),[.025,.975]).tolist()
    with (OUT/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'artifact_hashes.json', output_hashes)
    save(OUT/'video_inventory.json', video_rows)
    report = dict(status='PASS', episodes=2000, successes=sum(r['success'] for r in rows), success_rate=sum(r['success'] for r in rows)/2000,
                  suites=suite_reports, partitions=partitions, task_cluster_descriptive_95_interval=ci, total_steps=all_steps,total_calls=all_calls,
                  videos_verified=80, decoded_frames=sum(r['frames'] for r in video_rows), completed_results_reused=complete_reused,
                  interrupted_prefix_steps_retained=51, pending_actions_retained=4, next_call_step=55,
                  source_hashes_verified=len(sources), frozen_input_hashes_verified=len(plan['input_sha256']), selected_checkpoint_sha256=plan['checkpoint_sha256'],
                  initial_and_call_input_hashes='All2000 recorded initial hashes and every call hash checked for format and bound to immutable trace; original lossless RGB not retained, no posthoc full input-hash recomputation claimed.',
                  physical_validation='All stored states finite with consistent dimensions; all step actions match recorded queues exactly. Frozen runtime asserts done==env.check_success. Resumed161+51 steps physically checked in runner at1e-7; no new simulator replay in this audit.',
                  reset_validation='All four suite logs contain ordered task0..9/init0..49. Frozen source seeds7 once per suite/task environment and restores prior history with recorded actions; RNG state not explicitly serialized, no separate RNG-bitwise claim.',
                  elapsed_seconds=time.monotonic()-start,new_training_updates=0,new_rollouts=0,new_gpu_compute=False)
    assert report['successes']==aggregate['successes']
    save(OUT/'independent_acceptance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['suites']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
