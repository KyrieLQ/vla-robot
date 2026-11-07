"""Independent CPU check of M2 labels, checkpoints, queues, paired results and cost."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import imageio.v2 as imageio
from .data import sha256
from .model import ModelConfig
from .chunk_model_v1 import ChunkMiniVLA
from .experiment_v1 import ROOT

def main():
    started=time.monotonic();torch.set_num_threads(4)
    run=ROOT/'results/mini_vla_m2_20261019_run1';base=ROOT/'results/mini_vla_m1_20261019_run1'
    out=run/'review';out.mkdir(exist_ok=True)
    if (out/'independent_acceptance.json').exists():raise RuntimeError('Review already recorded')
    plan_path=ROOT/'configs/mini_vla_m2_20261019_v2.json';plan=json.loads(plan_path.read_text())
    assert sha256(base/'review/independent_acceptance.json')==plan['baseline_acceptance_sha256']
    source=json.loads((run/'source_manifest.json').read_text())
    for name,h in source.items():assert sha256(ROOT/name)==h,name
    history=json.loads((run/'training/history.json').read_text());training=json.loads((run/'training/report.json').read_text())
    assert len(history)==50 and [x['epoch'] for x in history]==list(range(1,51))
    for row in history:
        assert row['updates']==368*row['epoch'] and row['train_frames']==46982 and row['valid_target_steps']==452540
        assert row['validation']['frames']==5988 and row['validation']['valid_target_steps']==57720
    assert min(history,key=lambda x:x['validation']['normalized_mse'])['epoch']==training['best_epoch']==14
    cache=ROOT/plan['cache']/'validation';data={p.stem:np.load(p,mmap_mode='r') for p in cache.glob('*.npy')}
    ids=data['episode_id'];n=len(ids);assert n==5988
    # Independently construct future labels by grouping actual cached episode IDs.
    final_index={int(e):int(np.flatnonzero(ids==e)[-1]) for e in np.unique(ids)}
    targets=np.empty((n,10,7),dtype=np.float64);mask=np.zeros((n,10),dtype=bool)
    for i in range(n):
        end=final_index[int(ids[i])]
        for step in range(10):
            j=min(i+step,end);assert ids[j]==ids[i]
            targets[i,step]=data['actions'][j,0];mask[i,step]=i+step<=end
    assert mask.sum()==57720
    reference=json.loads((ROOT/plan['preprocessing']).read_text());checkpoints={};recomputed={}
    m1_initial=torch.load(base/'training/initial.pt',map_location='cpu',weights_only=False)
    for variant in ['initial','best','latest']:
        path=run/'training'/(variant+'.pt');assert sha256(path)==training['checkpoint_sha256'][variant]
        ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['plan_sha256']==sha256(plan_path) and ck['preprocessing']==reference
        model=ChunkMiniVLA(ModelConfig(**ck['model_config']));model.load_state_dict(ck['state_dict']);model.eval()
        assert sum(p.numel() for p in model.parameters())==794726
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if variant=='initial':
            for key,value in ck['state_dict'].items():
                if not key.startswith('action_head.2'):torch.testing.assert_close(value,m1_initial['state_dict'][key],rtol=0,atol=0)
        else:assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={ck['updates']}=={368*ck['epoch']}
        sums=np.zeros(7);all_sse=0.
        with torch.no_grad():
            for begin in range(0,n,64):
                end=min(begin+64,n);im=torch.tensor(np.array(data['images'][begin:end])).float()/255
                pred=model(image=im[:,0],wrist_image=im[:,1],state=torch.tensor(np.array(data['state'][begin:end])),
                           token_ids=torch.tensor(np.array(data['token_ids'][begin:end])),language_mask=torch.tensor(np.array(data['language_mask'][begin:end]))).numpy().astype('float64')
                errors=np.square(pred-targets[begin:end]);sums+=errors[:,0].sum(axis=0)
                all_sse+=float(np.sum(errors*mask[begin:end,:,None]))
        first=float(sums.sum()/(n*7));chunk=all_sse/(mask.sum()*7)
        recorded=json.loads((run/'training/initial_validation.json').read_text()) if variant=='initial' else history[ck['epoch']-1]['validation']
        assert abs(first-recorded['normalized_mse'])<2e-5 and abs(chunk-recorded['all_valid_chunk_mse'])<2e-5
        recomputed[variant]={'first_action_mse':first,'valid_chunk_mse':chunk,'first_action_recorded':recorded['normalized_mse'],'valid_target_steps':int(mask.sum())}
        checkpoints[variant]={'epoch':ck['epoch'],'updates':ck['updates'],'sha256':sha256(path)}
    expected={(t,i) for t in range(10) for i in [2,12,22,32,42]}
    sources={'m1':base/'best','h1':run/'h1','h5':run/'h5'};rows={};traces={};cost={};hashes={}
    for variant,directory in sources.items():
        rows[variant]={};traces[variant]={};latency=[];calls=0;steps=0;seconds=0.
        for ep in sorted(directory.glob('task*/episode.json')):
            row=json.loads(ep.read_text());key=(row['task_id'],row['initial_state_index']);assert key in expected and key not in rows[variant]
            path=ep.parent;assert sha256(path/'trace.npz')==row['trace_sha256'] and sha256(path/'external.mp4')==row['video_sha256']
            with np.load(path/'trace.npz') as z:trace={k:z[k] for k in z.files}
            assert trace['actions'].shape==(row['steps'],7) and np.isfinite(trace['actions']).all() and np.max(np.abs(trace['actions']))<=1
            assert len(trace['robot_state'])==row['steps'] and bool(trace['done'].any())==row['task_success'] and not trace['done'][:-1].any()
            assert row['task_success'] or row['steps']==220
            if variant!='m1':
                h=1 if variant=='h1' else 5
                assert row['checkpoint_sha256']==checkpoints['best']['sha256'] and row['plan_sha256']==sha256(plan_path)
                assert row['policy_calls']==(row['steps']+h-1)//h==len(trace['latency'])
                np.testing.assert_array_equal(trace['call_steps'],np.arange(0,row['steps'],h))
                assert trace['action_chunks'].shape==(row['policy_calls'],10,7)
                for step,action in enumerate(trace['actions']):np.testing.assert_array_equal(action,trace['action_chunks'][step//h,step%h])
            else:assert row['policy_calls']==row['steps']
            rows[variant][key]=row;traces[variant][key]=trace;latency.extend(trace['latency'][1:].tolist());calls+=row['policy_calls'];steps+=row['steps'];seconds+=float(trace['latency'].sum())
            hashes[str(path/'trace.npz')]=sha256(path/'trace.npz');hashes[str(path/'external.mp4')]=sha256(path/'external.mp4')
        assert set(rows[variant])==expected
        cost[variant]={'successes':sum(x['task_success'] for x in rows[variant].values()),'episodes':50,'calls':calls,'executed_steps':steps,
                       'rpc_seconds_sum':seconds,'rpc_median_ms':float(np.median(latency)*1000),'rpc_p95_ms':float(np.quantile(latency,.95)*1000),'calls_per_executed_step':calls/steps}
    comparisons={};render_differences={};physical_max=0.
    for name,a,b in [('chunk_supervision','m1','h1'),('execution_prefix','h1','h5'),('combined_vs_m1','m1','h5')]:
        delta=[];tasks=[];render_differences[name]=[]
        for key in sorted(expected):
            for field in ['initial_qpos','initial_qvel']:
                diff=float(np.max(np.abs(traces[a][key][field]-traces[b][key][field])));physical_max=max(physical_max,diff);assert diff<1e-7
            if rows[a][key]['initial_image_sha256']!=rows[b][key]['initial_image_sha256']:render_differences[name].append(key)
            delta.append(int(rows[b][key]['task_success'])-int(rows[a][key]['task_success']))
        for task in range(10):tasks.append(np.mean([int(rows[b][k]['task_success'])-int(rows[a][k]['task_success']) for k in expected if k[0]==task]))
        rng=np.random.default_rng(20261019);boot=np.array(tasks)[rng.integers(0,10,(10000,10))].mean(axis=1)
        comparisons[name]={'wins':delta.count(1),'losses':delta.count(-1),'difference':float(np.mean(delta)),'interval':np.quantile(boot,[.025,.975]).tolist()}
    original=json.loads((run/'analysis.json').read_text())
    for name,value in comparisons.items():
        assert value['wins']==original['comparisons'][name]['wins'] and value['losses']==original['comparisons'][name]['losses']
        np.testing.assert_allclose(value['interval'],original['comparisons'][name]['task_cluster_descriptive_95_interval'])
    h5_rescue=sorted(k for k in expected if not rows['h1'][k]['task_success'] and rows['h5'][k]['task_success'])[0]
    h5_harm=sorted(k for k in expected if rows['h1'][k]['task_success'] and not rows['h5'][k]['task_success'])[0]
    remaining=sorted(k for k in expected if not rows['h5'][k]['task_success'])[0]
    selections=[('h5',(3,2)),('h1',h5_rescue),('h5',h5_rescue),('h1',h5_harm),('h5',h5_harm),('h5',remaining)]
    frames=[]
    for variant,key in dict.fromkeys(selections):
        row=rows[variant][key];case=row['pair_id'];reader=imageio.get_reader(sources[variant]/case/'external.mp4')
        for step in sorted(set([0,row['steps']//3,2*row['steps']//3,row['steps']])):
            path=out/('%s_%s_frame%03d.png'%(variant,case,step));imageio.imwrite(path,reader.get_data(step))
            frames.append({'variant':variant,'pair_id':case,'success':row['task_success'],'frame':step,'path':str(path.relative_to(ROOT))})
        reader.close()
    result={'status':'PASS','source_hash_count':len(source),'new_episodes':100,'reused_m1_episodes':50,'validation_recalculation':recomputed,'checkpoints':checkpoints,
            'cost':cost,'comparisons':comparisons,'initial_physics_max_error':physical_max,'render_differences':render_differences,'executed_queue':'EXACT_ALL_100',
            'h5_total_call_reduction_vs_h1':1-cost['h5']['calls']/cost['h1']['calls'],
            'h5_calls_per_step_reduction_vs_h1':1-cost['h5']['calls_per_executed_step']/cost['h1']['calls_per_executed_step'],
            'train_best_epoch':history[13]['train_mse'],'first_val_best_epoch':history[13]['validation']['normalized_mse'],
            'train_last_epoch':history[-1]['train_mse'],'first_val_last_epoch':history[-1]['validation']['normalized_mse'],
            'video_frames':frames,'artifact_hashes':hashes,'elapsed_seconds':time.monotonic()-started,
            'boundary':'single seed and reused development starts; no independent confirmation; auxiliary future labels and 8127 parameters are part of chunk treatment'}
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','cost','comparisons','elapsed_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
