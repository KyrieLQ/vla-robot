"""Bounded CPU diagnosis of existing Flow artifacts; never updates any model."""
import hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from mini_vla.vlm_action_heads_v1 import FlowHead, RegressionHead, HeadConfig

R = Path(__file__).resolve().parents[1]
OUT = R / 'reports/flow_diagnosis_20261103'
RUN = R / 'results/mini_vla_m7_fixed_lora_flow_20261029_run1'
REG = R / 'results/mini_vla_m6_paired_20261028_run1/lora'
START = time.monotonic()

def write(name, value):
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(OUT / name)

def log(stage):
    print(stage, flush=True)
    write('status.json', dict(status='RUNNING', stage=stage, pid=os.getpid(), elapsed_seconds=time.monotonic()-START))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(pred, target, mask):
    pred, target, mask = np.asarray(pred), np.asarray(target), np.asarray(mask)
    error = (pred.astype(np.float64)-target)**2
    return dict(first_mse=float(error[:,0].mean()),
                first_per_dim=error[:,0].mean(0).tolist(),
                full_window_first_mse=float(error[mask.all(1),0].mean()) if mask.all(1).any() else None,
                tail_first_mse=float(error[~mask.all(1),0].mean()) if (~mask.all(1)).any() else None,
                valid_chunk_mse=float(error[mask].mean()),
                horizon_mse=[float(error[:,k][mask[:,k]].mean()) for k in range(10)])

def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    p = json.loads((RUN/'plan.json').read_text())
    c = torch.load(RUN/'training/best.pt', map_location='cpu', weights_only=False)
    rc = torch.load(REG/'endpoint.pt', map_location='cpu', weights_only=False)
    flow = FlowHead(HeadConfig(**c['model_config'])).eval().requires_grad_(False)
    flow.load_state_dict(c['state_dict'])
    reg = RegressionHead(HeadConfig(**rc['model_config'])).eval().requires_grad_(False)
    reg.load_state_dict(rc['state_dict'])
    result = dict(plan_sha256=sha(OUT/'plan.json'), script_sha256=sha(Path(__file__)),
                  flow_sha256=sha(RUN/'training/best.pt'), regression_sha256=sha(REG/'endpoint.pt'),
                  probe_scope='120 existing TRAIN inputs; not validation or new closed-loop score',
                  new_optimizer_updates=0,new_simulation_episodes=0, device='cpu')
    log('Full saved validation predictions')
    index = json.loads((R/p['cache']/'validation/index.json').read_text())
    actions = np.load(R/p['numeric_cache']/'validation/actions.npy', mmap_mode='r')[:,0]
    n = len(actions)
    target = np.zeros((n,10,7),np.float32); mask = np.zeros((n,10),bool)
    suites = np.empty(n,object)
    for e in index:
        ids = np.arange(e['start'],e['end'])[:,None]+np.arange(10)[None]
        mask[e['start']:e['end']] = ids < e['end']
        target[e['start']:e['end']] = actions[np.minimum(ids,e['end']-1)]
        suites[e['start']:e['end']] = e['suite']
    saved = {}
    for name,path in [('flow',RUN/'training/best_predictions.npy'),('regression',REG/'endpoint_predictions.npy')]:
        pred=np.load(path)
        m=metrics(pred,target,mask)
        m['by_suite']={s:metrics(pred[suites==s],target[suites==s],mask[suites==s]) for s in sorted(set(suites))}
        m['tail_first_mse']=float(((pred[~mask.all(1),0]-target[~mask.all(1),0])**2).mean())
        m['full_window_first_mse']=float(((pred[mask.all(1),0]-target[mask.all(1),0])**2).mean())
        norm=c['preprocessing']['normalization']['actions']
        raw=pred*np.asarray(norm['scale'])+np.asarray(norm['mean'])
        gt=target*np.asarray(norm['scale'])+np.asarray(norm['mean'])
        m['first_raw_per_dim_mse']=((raw[:,0]-gt[:,0])**2).mean(0).tolist()
        m['first_gripper_sign_accuracy']=float(((raw[:,0,6]>=0)==(gt[:,0,6]>=0)).mean())
        m['first_clip_fraction_by_dim']=(np.abs(raw[:,0])>1).mean(0).tolist()
        saved[name]=m
    result['full_validation']=dict(frames=n,tail_frames=int((~mask.all(1)).sum()),models=saved)
    h=json.loads((RUN/'training/history.json').read_text())
    result['history']=[{k:row[k] for k in ['epoch','train_velocity_mse']}|dict(validation_mse=row['validation']['normalized_mse']) for row in h]
    result['train_tail_frame_fraction']=sum(min(9,e['end']-e['start']) for e in json.loads((R/p['cache']/'train/index.json').read_text()))/243000
    write('results.json',result)
    log('Load persisted adapted gate120 features; no VLM forward')
    samples=json.loads((R/p['sample_manifest']).read_text())['samples']
    gate=json.loads((RUN/'frontend_gate.json').read_text())['rows']
    ix={e['episode_id']:e for e in json.loads((R/p['cache']/'train/index.json').read_text())}
    nums={k:np.load(R/p['numeric_cache']/'train'/(k+'.npy'),mmap_mode='r') for k in ['actions','state']}
    L=max(ix[s['episode_id']]['tokens'] for s in samples)
    hidden=torch.zeros(120,L,960); hm=torch.zeros(120,L,dtype=torch.bool)
    state=[]; targets=[]; masks=[]
    for i,(s,g) in enumerate(zip(samples,gate)):
        assert (s['episode_id'],s['frame_index'])==(g['episode_id'],g['frame_index'])
        f=RUN/'gate'/g['file']; assert sha(f)==g['file_sha256']
        a=torch.from_numpy(np.load(f)).view(torch.bfloat16).float()
        hidden[i,:len(a)]=a; hm[i,:len(a)]=True
        e=ix[s['episode_id']]; start=e['start']+s['frame_index']; ids=start+np.arange(10)
        masks.append(ids<e['end']); targets.append(nums['actions'][np.minimum(ids,e['end']-1),0]);state.append(nums['state'][start])
    y=torch.tensor(np.array(targets)); mask=torch.tensor(np.array(masks));state=torch.tensor(np.array(state))
    obs=dict(hidden=hidden,hidden_mask=hm,state=state)
    with torch.inference_mode():
        mem,pad=flow.memory(**obs)
        rp=reg(**obs)
    def velocity(x,t,mm=mem,pp=pad,am=None):
        return flow.velocity(memory=mm,memory_padding=pp,noisy_action=x,time=torch.full((120,),float(t)),action_mask=am)
    @torch.inference_mode()
    def sample(z,steps=10,heun=False,mm=mem,pp=pad,am=None):
        x=z.clone()
        for i in range(steps):
            v=velocity(x,i/steps,mm,pp,am)
            if heun:v=(v+velocity(x+v/steps,(i+1)/steps,mm,pp,am))/2
            x=x+v/steps
        return x
    z=torch.randn((120,10,7),generator=torch.Generator().manual_seed(20261103))
    with torch.inference_mode():
        torch.testing.assert_close(sample(z),flow.sample(**obs,noise=z),atol=0,rtol=0)
    probe=dict(samples=120,tasks=len(set((s['suite'],s['task_id']) for s in samples)),tail_samples=int((~mask.all(1)).sum()),
               regression=metrics(rp.numpy(),y.numpy(),mask.numpy()),solvers={})
    predictions={}
    for steps in [5,10,20,40,100]:
        log(f'CPU fixed-noise Euler{steps}, 120 training inputs')
        pred=sample(z,steps);predictions[f'euler{steps}']=pred
        probe['solvers'][f'euler{steps}']=metrics(pred.numpy(),y.numpy(),mask.numpy())
    pred=sample(z,10,True);predictions['heun10']=pred
    probe['solvers']['heun10']=metrics(pred.numpy(),y.numpy(),mask.numpy())
    for name,pred in predictions.items():
        probe['solvers'][name]['first_mse_vs_euler100']=float(((pred[:,0]-predictions['euler100'][:,0])**2).mean())
    log('Four fixed seeds, teacher-path and condition sensitivity')
    noise_samples=[predictions['euler10']]
    for seed in [20261104,20261105,20261106]:
        zz=torch.randn((120,10,7),generator=torch.Generator().manual_seed(seed))
        noise_samples.append(sample(zz))
    stack=torch.stack(noise_samples)
    avg=stack.mean(0)
    probe['multi_noise']=dict(single_seed_first_mse=[float(((v[:,0]-y[:,0])**2).mean()) for v in stack],
        four_sample_mean=metrics(avg.numpy(),y.numpy(),mask.numpy()),
        first_sample_variance=float(stack[:,:,0].var(0,unbiased=False).mean()),
        sample_variance_per_dim=stack[:,:,0].var(0,unbiased=False).mean(0).tolist())
    with torch.inference_mode():
        teacher=[]
        for t in [0,.05,.1,.2,.4,.6,.8,.9,.99]:
            x=(1-t)*z+t*y
            v=velocity(x,t)
            vm=velocity(x,t,am=mask)
            err=(v-(y-z)).square()
            teacher.append(dict(t=t,velocity_mse=float(err[mask].mean()),first_velocity_mse=float(err[:,0].mean()),
                full_window_first_velocity_mse=float(err[mask.all(1),0].mean()),
                masked_first_velocity_mse=float(((vm-(y-z))[:,0]**2).mean()),
                reconstructed_first_mse=float(((x+(1-t)*v-y)[:,0]**2).mean()),
                first_velocity_mask_difference_mse=float(((v-vm)[:,0]**2).mean())))
        probe['teacher_path']=teacher
        perm=torch.randperm(120,generator=torch.Generator().manual_seed(123))
        shuf=sample(z,mm=mem[perm],pp=pad[perm])
        probe['shuffled_observation']=metrics(shuf.numpy(),y.numpy(),mask.numpy())
        handle=flow.time_mlp.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
        no_time=sample(z);handle.remove()
        probe['zero_time_embedding']=metrics(no_time.numpy(),y.numpy(),mask.numpy())
        masked=sample(z,am=mask)
        probe['teacher_tail_mask_sampling']=metrics(masked.numpy(),y.numpy(),mask.numpy())
        qnoise=flow.noisy_action_projection(z);qpos=flow.action_queries
        embedding=[]
        for t in [0,.1,.5,.9,1]:
            phase=t*1000*flow.time_frequencies
            emb=flow.time_mlp(torch.cat((phase.sin(),phase.cos())))
            embedding.append(dict(t=t,time_rms=float(emb.square().mean().sqrt())))
        probe['input_scales']=dict(noisy_action_projection_rms=float(qnoise.square().mean().sqrt()),query_rms=float(qpos.square().mean().sqrt()),time=embedding)
        # Exact bias/variance identity is a consistency check, NOT independent evidence.
        lhs=np.mean(probe['multi_noise']['single_seed_first_mse'])
        rhs=probe['multi_noise']['four_sample_mean']['first_mse']+probe['multi_noise']['first_sample_variance']
        assert abs(lhs-rhs)<1e-6
    result['probe']=probe
    result['elapsed_seconds']=time.monotonic()-START
    write('results.json',result)
    write('status.json',dict(status='COMPLETE',elapsed_seconds=result['elapsed_seconds'],pid=os.getpid(),new_optimizer_updates=0,new_simulation_episodes=0))
    print(json.dumps(dict(full_validation=saved,probe=probe,elapsed=result['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        write('status.json',dict(status='FAILED',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START))
        raise
