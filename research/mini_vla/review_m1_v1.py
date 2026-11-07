"""Independent CPU recomputation of M1 artifacts; does not train or simulate."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio
from .model import MiniVLA,ModelConfig

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.monotonic();torch.set_num_threads(4)
    run=ROOT/'results/mini_vla_m1_20261019_run1';out=run/'review';out.mkdir(exist_ok=True)
    if (out/'independent_acceptance.json').exists():raise RuntimeError('Review already recorded')
    plan_path=ROOT/'configs/mini_vla_m1_20261019_v2.json';plan=json.loads(plan_path.read_text())
    hashes=json.loads((run/'source_manifest.json').read_text())
    for name,value in hashes.items():assert sha(ROOT/name)==value,name
    training=json.loads((run/'training/report.json').read_text());hist=json.loads((run/'training/history.json').read_text())
    assert [x['epoch'] for x in hist]==list(range(1,51))
    assert all(x['train_frames']==46982 and x['validation']['frames']==5988 and x['updates']==368*x['epoch'] for x in hist)
    chosen=min(hist,key=lambda x:x['validation']['normalized_mse'])
    assert chosen['epoch']==training['best_epoch']==9
    split=json.loads((ROOT/plan['split']).read_text());pre=json.loads((ROOT/plan['preprocessing']).read_text())
    assert not {e['episode_id'] for e in split['train']} & {e['episode_id'] for e in split['validation']}
    assert set(pre['fit_episode_ids'])=={e['episode_id'] for e in split['train']}
    cache=ROOT/plan['cache']/'validation'
    arrays={p.stem:np.load(p,mmap_mode='r') for p in cache.glob('*.npy')};n=len(arrays['state']);assert n==5988
    recomputed={};checkpoint_checks={}
    for variant in ['initial','best','latest']:
        path=run/'training'/(variant+'.pt');assert sha(path)==training['checkpoint_sha256'][variant]
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['plan_sha256']==sha(plan_path) and ck['preprocessing']==pre
        if variant!='initial':
            steps={int(s['step']) for s in ck['optimizer']['state'].values()}
            assert steps=={ck['updates']} and ck['updates']==ck['epoch']*368
        model=MiniVLA(ModelConfig(**ck['model_config']));model.load_state_dict(ck['state_dict']);model.eval()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        sums=np.zeros(7,dtype='float64');reference=0.
        with torch.no_grad():
            for begin in range(0,n,64):
                end=min(begin+64,n);im=torch.tensor(np.array(arrays['images'][begin:end],copy=True)).float()/255
                output=model(image=im[:,0],wrist_image=im[:,1],state=torch.tensor(np.array(arrays['state'][begin:end])),
                             token_ids=torch.tensor(np.array(arrays['token_ids'][begin:end])),language_mask=torch.tensor(np.array(arrays['language_mask'][begin:end])))
                target=arrays['actions'][begin:end].astype('float64')
                sums+=np.square(output.numpy().astype('float64')-target).sum(axis=(0,1));reference+=np.square(target).sum()
        value=float(sums.sum()/(n*7));recorded=(json.loads((run/'training/initial_validation.json').read_text())['normalized_mse'] if variant=='initial' else hist[ck['epoch']-1]['validation']['normalized_mse'])
        assert abs(value-recorded)<2e-5,(variant,value,recorded)
        recomputed[variant]={'frames':n,'cpu_normalized_mse':value,'recorded_mse':recorded,'absolute_difference':abs(value-recorded),'per_dimension_mse':(sums/n).tolist()}
        checkpoint_checks[variant]={'epoch':ck['epoch'],'updates':ck['updates'],'sha256':sha(path),'parameters':sum(p.numel() for p in model.parameters())}
    expected={(task,init) for task in plan['evaluation']['task_ids'] for init in plan['evaluation']['initial_state_indices']}
    outcomes={};traces={};latencies={};artifacts={};motion=[]
    for variant in ['initial','best']:
        outcomes[variant]={};traces[variant]={};latencies[variant]=[]
        for episode in sorted((run/variant).glob('task*/episode.json')):
            d=json.loads(episode.read_text());key=(d['task_id'],d['initial_state_index']);assert key in expected and key not in outcomes[variant]
            trace_file=episode.parent/'trace.npz';video=episode.parent/'external.mp4'
            assert d['plan_sha256']==sha(plan_path) and d['checkpoint_sha256']==checkpoint_checks[variant]['sha256']
            assert sha(trace_file)==d['trace_sha256'] and sha(video)==d['video_sha256'] and video.stat().st_size>1000
            with np.load(trace_file) as z:t={k:z[k] for k in z.files}
            assert t['actions'].shape==(d['steps'],7) and t['robot_state'].shape==(d['steps'],8)
            assert np.isfinite(t['actions']).all() and np.max(np.abs(t['actions']))<=1
            assert len(t['done'])==len(t['latency'])==d['steps']==d['policy_calls']
            assert bool(t['done'].any())==d['task_success'] and not t['done'][:-1].any()
            assert d['task_success'] or d['steps']==220
            outcomes[variant][key]=d;traces[variant][key]=t;latencies[variant].extend(t['latency'][1:].tolist())
            artifacts[str(trace_file.relative_to(run))]=sha(trace_file);artifacts[str(video.relative_to(run))]=sha(video)
            pos=t['robot_state'][:,:3];tail=pos[-30:]
            motion.append({'variant':variant,'pair_id':d['pair_id'],'success':d['task_success'],
                           'tail_position_span_m':float(np.linalg.norm(np.ptp(tail,axis=0))),
                           'tail_path_length_m':float(np.linalg.norm(np.diff(tail,axis=0),axis=1).sum()),
                           'action_change_l2_mean':float(np.linalg.norm(np.diff(t['actions'],axis=0),axis=1).mean())})
        assert set(outcomes[variant])==expected
    paired=[];per_task=[];physical_max=0.;image_diffs=[]
    for key in sorted(expected):
        a,b=outcomes['initial'][key],outcomes['best'][key]
        for field in ['initial_qpos','initial_qvel']:
            difference=float(np.max(np.abs(traces['initial'][key][field]-traces['best'][key][field])))
            physical_max=max(physical_max,difference);assert difference<1e-7
        if a['initial_image_sha256']!=b['initial_image_sha256']:
            external_a=np.array(Image.open(run/'initial'/a['pair_id']/'first.png')).astype('int16')
            external_b=np.array(Image.open(run/'best'/b['pair_id']/'first.png')).astype('int16')
            diff=np.abs(external_a-external_b)
            image_diffs.append({'pair_id':a['pair_id'],'external_changed_channels':int(np.count_nonzero(diff)),
                                'external_max_uint8_difference':int(diff.max()),'wrist_difference':'not saved separately; cannot localize combined hash mismatch there'})
        paired.append(int(b['task_success'])-int(a['task_success']))
    for task in range(10):
        keys=sorted(k for k in expected if k[0]==task)
        per_task.append({'task':task,'initial':sum(outcomes['initial'][k]['task_success'] for k in keys),
                         'trained':sum(outcomes['best'][k]['task_success'] for k in keys),'n':len(keys)})
    task_diff=np.array([(r['trained']-r['initial'])/r['n'] for r in per_task])
    rng=np.random.default_rng(20261019);boot=np.mean(task_diff[rng.integers(0,10,size=(10000,10))],axis=1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    original=json.loads((run/'analysis.json').read_text())
    assert sum(paired)==original['paired_wins']-original['paired_losses']==18
    np.testing.assert_allclose(ci,original['task_cluster_descriptive_95_interval'])
    # Fixed illustrative sample: both strong and weak tasks, not exhaustive failure annotation.
    selected=['task6_init2','task0_init12','task0_init2','task3_init2','task8_init2']
    frames=[]
    for case in selected:
        path=run/'best'/case;row=json.loads((path/'episode.json').read_text())
        reader=imageio.get_reader(path/'external.mp4')
        for t in sorted(set([0,row['steps']//3,2*row['steps']//3,row['steps']])):
            target=out/(case+'_frame%03d.png'%t);imageio.imwrite(target,reader.get_data(t))
            frames.append({'case':case,'frame':t,'path':str(target.relative_to(ROOT))})
        reader.close()
    result={'status':'PASS','source_files_verified':len(hashes),'paired_episodes_verified':100,'unique_cases':50,
            'checkpoints':checkpoint_checks,'validation_recalculation':recomputed,'mean_action_reference_mse':reference/(n*7),
            'per_task':per_task,'successes':{'initial':0,'trained':18},'paired_wins':paired.count(1),'paired_losses':paired.count(-1),
            'task_cluster_descriptive_95_interval':ci,'initial_physics_max_error':physical_max,'render_differences':image_diffs,
            'rpc_latency_seconds':{k:{'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'n':len(v)} for k,v in latencies.items()},
            'train_epoch9':hist[8]['train_mse'],'validation_epoch9':hist[8]['validation']['normalized_mse'],
            'train_epoch50':hist[-1]['train_mse'],'validation_epoch50':hist[-1]['validation']['normalized_mse'],
            'motion_descriptors':motion,'video_sample_frames':frames,'artifact_sha256':artifacts,
            'evidence_boundary':'single training seed and 50 known-task development starts; sample video descriptions are not full failure attribution',
            'elapsed_seconds':time.monotonic()-started}
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','paired_episodes_verified','successes','validation_recalculation','elapsed_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
