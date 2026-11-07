"""Existing training inputs only: exact frontend/cache identity and deployed action contract."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from .policy_m9_v1 import R, load_policy, sha
from .vlm_frontend_v1 import decode_record, bits_bf16


@torch.inference_mode()
def check(plan, output, policy=None):
    started = time.monotonic()
    if policy is None:
        policy, _ = load_policy(plan)
    samples = json.loads((R/plan['sample_manifest']).read_text())['samples']
    split = json.loads((R/plan['split']).read_text())
    allowed = {e['episode_id'] for e in split['train']}
    forbidden = {e['episode_id'] for e in split['validation']}
    assert len(samples) == 120 and all(s['episode_id'] in allowed-forbidden for s in samples)
    cache = R/plan['cache']
    index = {e['episode_id']: e for e in json.loads((cache/'train/index.json').read_text())}
    state_cache = np.load(R/plan['numeric_cache']/'train/state.npy', mmap_mode='r')
    reports = []
    latencies = []
    for s in samples:
        eid, frame = s['episode_id'], s['frame_index']
        source = R/plan['dataset_root']/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
        row = pq.read_table(source, columns=['image','wrist_image','state','episode_index','frame_index'], use_threads=False).slice(frame, 1).to_pylist()[0]
        assert row['episode_index'] == eid and row['frame_index'] == frame
        record = decode_record(row, s['instruction'])
        obs = {'observation/image':np.asarray(record['images'][0]),
               'observation/wrist_image':np.asarray(record['images'][1]),
               'observation/state':np.asarray(row['state']), 'prompt':s['instruction'],
               'sampling/task_id':s['task_id'],'sampling/initial_state_index':2,'sampling/call_index':frame}
        tick = time.monotonic()
        deployed = policy.infer(obs)
        latencies.append(time.monotonic()-tick)
        entry = index[eid]; folder = cache/entry['directory']
        stored = np.load(folder/'hidden.npy', mmap_mode='r')[frame]
        hashes = json.loads((folder/'input_hashes.json').read_text())
        expected_hash = hashes[frame]
        if isinstance(expected_hash, dict):
            expected_hash = expected_hash['input_sha256']
        assert deployed['vlm_input_sha256'] == expected_hash
        assert deployed['hidden_sha256'] == hashlib.sha256(stored.tobytes()).hexdigest()
        h = bits_bf16(stored)[None].cuda()
        st = torch.from_numpy(np.array(state_cache[entry['start']+frame]))[None].cuda()
        seed=int.from_bytes(hashlib.sha256(f"20261020:{s['task_id']}:2:{frame}".encode()).digest()[:8],'little')%(2**63-1)
        noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
        assert deployed['noise_seed']==seed and deployed['noise_sha256']==hashlib.sha256(noise.numpy().tobytes()).hexdigest()
        memory,padding=policy.head.memory(hidden=h,hidden_mask=torch.ones(h.shape[:2],dtype=torch.bool,device=h.device),state=st)
        generated=noise[None].cuda()
        for step in range(10):
            tau=torch.full((1,),step/10,dtype=torch.float32,device=h.device)
            generated=generated+policy.head.velocity(memory=memory,memory_padding=padding,noisy_action=generated,time=tau)/10
        expected=generated[0].cpu().numpy()
        np.testing.assert_allclose(deployed['normalized_actions'], expected, atol=1e-6, rtol=0)
        stats = policy.normalization['actions']
        command = np.clip(expected*np.asarray(stats['scale'],dtype=np.float32)+np.asarray(stats['mean'],dtype=np.float32),-1,1)
        np.testing.assert_allclose(deployed['actions'], command, atol=1e-6, rtol=0)
        reports.append(dict(episode_id=eid,frame_index=frame,input_sha256=expected_hash,
                            normalized_action_max_error=float(np.max(np.abs(deployed['normalized_actions']-expected))),
                            command_max_error=float(np.max(np.abs(deployed['actions']-command)))))
    result = dict(status='PASS',samples=120,all_features_bitwise_equal=True,
                  action_atol=1e-6,maximum_action_error=max(x['normalized_action_max_error'] for x in reports),
                  model_loading=policy.frontend.loading,noise_seed_and_ten_step_Euler_verified=True,head_parameters=sum(p.numel() for p in policy.head.parameters()),
                  frozen_vlm_parameters=sum(p.numel() for p in policy.frontend.model.parameters()),
                  frontend_trainable_parameters=sum(p.numel() for p in policy.frontend.model.parameters() if p.requires_grad),
                  inference_seconds_median=float(np.median(latencies)),inference_seconds_p95=float(np.quantile(latencies,.95)),
                  elapsed_seconds=time.monotonic()-started,checkpoint_sha256=plan['checkpoint_sha256'],
                  adapter_sha256=sha(R/'mini_vla/policy_m9_v1.py'),new_simulation_episodes=0,new_optimizer_updates=0,rows=reports)
    assert result['head_parameters']==432711 and result['frozen_vlm_parameters']==507482304 and result['frontend_trainable_parameters']==0
    output=Path(output);output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    check(json.loads(Path(a.plan).read_text()),a.output)
