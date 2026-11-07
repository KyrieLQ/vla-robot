"""Complete fixed50 comparison, trace verification and descriptive task-cluster uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .pipeline_v1 import ROOT,write

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run);rows={};traces={};metrics={};artifacts={}
    assert json.loads((run/'online_contract.json').read_text())['status']=='PASS'
    assert json.loads((run/'simulation_preflight/result.json').read_text())['status']=='PASS'
    assert sha(ROOT/plan['baseline_acceptance'])==plan['baseline_acceptance_sha256']
    torch.set_num_threads(4)
    assert sha(ROOT/plan['m8_baseline_acceptance'])==plan['m8_baseline_acceptance_sha256']
    expected={(t,i) for t in range(10) for i in [2,12,22,32,42]}
    for name,directory in [('m6',ROOT/plan['baseline_evaluation']),('m8',ROOT/plan['m8_baseline_evaluation']),('m9',run/'h5')]:
        result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS' and len(result['episodes'])==50
        rows[name]={};traces[name]={};latencies=[];server=[]
        for e in result['episodes']:
            k=(e['task_id'],e['initial_state_index']);assert k in expected and k not in rows[name]
            folder=directory/e['pair_id'];assert sha(folder/'trace.npz')==e['trace_sha256'] and sha(folder/'external.mp4')==e['video_sha256']
            assert json.loads((folder/'episode.json').read_text())==e
            with np.load(folder/'trace.npz') as z:t={f:z[f] for f in z.files}
            steps=e['steps'];assert steps==len(t['actions'])==len(t['done']) and e['policy_calls']==len(t['call_steps'])
            assert e['task_success']==bool(t['done'].any()) and not t['done'][:-1].any()
            assert e['task_success'] or steps==220
            assert np.isfinite(t['actions']).all() and np.max(np.abs(t['actions']))<=1.000001
            np.testing.assert_array_equal(t['call_steps'],np.arange(0,steps,5))
            np.testing.assert_array_equal(t['action_chunks'][:,:5].reshape(-1,7)[:steps],t['actions'])
            if name=='m9':
                assert e['checkpoint_sha256']==plan['checkpoint_sha256'] and e['plan_sha256']==sha(ROOT/plan.get('scientific_plan',str(Path(a.plan).resolve())))
                assert e['euler_steps']==10 and len(t['noise_seed'])==len(t['call_steps'])
                for call,(saved_seed,saved_hash) in enumerate(zip(t['noise_seed'],t['noise_sha256'])):
                    seed=int.from_bytes(hashlib.sha256(f"20261020:{k[0]}:{k[1]}:{call}".encode()).digest()[:8],'little')%(2**63-1)
                    assert int(saved_seed)==seed
                    noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
                    assert saved_hash==hashlib.sha256(noise.numpy().tobytes()).hexdigest()
                assert len(t['observation_sha256'])==len(t['vlm_input_sha256'])==len(t['call_steps'])
                stats=json.loads((ROOT/plan['preprocessing']).read_text())['normalization']['actions']
                recovered=np.clip(t['normalized_action_chunks']*np.asarray(stats['scale'],dtype=np.float32)+np.asarray(stats['mean'],dtype=np.float32),-1,1)
                np.testing.assert_allclose(t['action_chunks'],recovered,atol=1e-6,rtol=0)
            rows[name][k]=e;traces[name][k]=t;latencies.extend(t['latency'][1:]);server.extend(t['server_ms'][1:])
            artifacts[str(folder/'episode.json')]=sha(folder/'episode.json')
        assert set(rows[name])==expected
        metrics[name]=dict(successes=sum(e['task_success'] for e in rows[name].values()),episodes=50,
                           policy_calls=sum(e['policy_calls'] for e in rows[name].values()),
                           steps=sum(e['steps'] for e in rows[name].values()),elapsed_seconds=result['elapsed_seconds'],
                           rpc_median_ms=float(np.median(latencies)*1000),rpc_p95_ms=float(np.quantile(latencies,.95)*1000),
                           server_median_ms=float(np.median(server)),server_p95_ms=float(np.quantile(server,.95)))
    pairs=[];render=[];bytask=[]
    for k in sorted(expected):
        for f in ('initial_qpos','initial_qvel'):np.testing.assert_allclose(traces['m6'][k][f],traces['m9'][k][f],atol=1e-7,rtol=0)
        for f in ('initial_qpos','initial_qvel'):np.testing.assert_allclose(traces['m8'][k][f],traces['m9'][k][f],atol=1e-7,rtol=0)
        assert rows['m6'][k]['language']==rows['m8'][k]['language']==rows['m9'][k]['language']
        if rows['m6'][k]['initial_image_sha256']!=rows['m9'][k]['initial_image_sha256']:render.append(list(k))
        pairs.append(dict(task_id=k[0],initial_state_index=k[1],m6=bool(rows['m6'][k]['task_success']),m8=bool(rows['m8'][k]['task_success']),m9=bool(rows['m9'][k]['task_success']),difference=int(rows['m9'][k]['task_success'])-int(rows['m6'][k]['task_success'])))
    for t in range(10):bytask.append(float(np.mean([p['difference'] for p in pairs if p['task_id']==t])))
    b=np.random.default_rng(20261019).choice(bytask,(10000,10),replace=True).mean(1);diff=[p['difference'] for p in pairs]
    diff8=[int(p['m9'])-int(p['m8']) for p in pairs]
    tasks8=[float(np.mean([int(p['m9'])-int(p['m8']) for p in pairs if p['task_id']==t])) for t in range(10)]
    boot8=np.random.default_rng(20261019).choice(tasks8,(10000,10),replace=True).mean(1)
    paired8=dict(wins=diff8.count(1),losses=diff8.count(-1),difference=float(np.mean(diff8)),task_cluster_descriptive_95_interval=np.quantile(boot8,[.025,.975]).tolist())
    report=dict(paired_m8=paired8,status='COMPLETE_REVIEW_REQUIRED',metrics=metrics,
                paired=dict(wins=diff.count(1),losses=diff.count(-1),difference=float(np.mean(diff)),task_cluster_descriptive_95_interval=np.quantile(b,[.025,.975]).tolist()),
                initial_render_mismatch=render,pairs=pairs,artifacts_sha256=artifacts,
                new_optimizer_updates=0,confirmation_accessed=False,
                claim_boundary='AdaptiveSpatial50 pilot,one training seed. M9flow versusM8regression sharesVLM/data but changesobjective,headparameters and generation;M6 differs in frontend too;not isolatedVLM causalablation,not fullLIBERO/generalization score.')
    write(run/'analysis.json',report);print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
