"""One full adapted feature extraction; matched complete regression optimization.

Try two independent GPU optimizers in one process sharing one immutable RAM
cache. Use the measured faster admitted mode. No DDP, no sample/batch changes.
"""
import argparse,copy,gc,hashlib,json,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .regression_depth_pair_v1 import RegressionDepthHead,verify_initial
from .vlm_action_heads_v1 import HeadConfig
from .model import masked_mse
from .policy_paired_v1 import AdaptedFrontend
from .ram_token_data_v1 import RamTokenData
from .vlm_cached_data_v2 import collate_cached
from .train_vlm_head_v1 import seed,optimizer,loader,sha,write,save
from .train_flow_tail_consistent_ram_v1 import frontend_gate,token_sha
from .adapted_feature_parallel_v2 import extract
from flow_ram_resources_v1 import admission,guard
R=Path(__file__).resolve().parents[1]
ARMS=('depth1','depth2')

def initial_models(p):
    models={}
    for name,layers in zip(ARMS,(1,2)):
        seed(42);models[name]=RegressionDepthHead(HeadConfig(**p['head_config']),layers)
    ck=torch.load(R/p['regression_initial'],map_location='cpu',weights_only=False)
    verify_initial(models['depth1'],models['depth2'],ck['state_dict'])
    return models

def move_to(batch,device):
    obs={k:batch[k].to(device,non_blocking=True) for k in ['hidden','hidden_mask','state']}
    return obs,batch['actions'].to(device,non_blocking=True),batch['action_mask'].to(device,non_blocking=True)

def synchronize(devices):
    for dev in sorted(set(devices)):torch.cuda.synchronize(dev)

def optimize_batch(models,opts,devices,batch):
    losses={}
    for name in ARMS:
        obs,target,mask=move_to(batch,devices[name]);opts[name].zero_grad(set_to_none=True)
        pred=models[name](**obs);losses[name]=masked_mse(pred,target,mask)
    for name in ARMS:
        losses[name].backward();torch.nn.utils.clip_grad_norm_(models[name].parameters(),1.,error_if_nonfinite=True);opts[name].step()
    values={name:float(loss.detach()) for name,loss in losses.items()}
    if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Nonfinite training loss')
    return values

@torch.no_grad()
def evaluate_pair(models,devices,data,run,stage,export=False):
    totals={n:dict(first=0.,chunk=0.,constant=0.,per={i:[0.,0] for i in range(40)}) for n in ARMS};frames=targets=0
    predictions={n:np.empty((len(data),10,7),np.float32) for n in ARMS} if export else None
    for model in models.values():model.eval()
    for batch in loader(data,workers=0):
        outputs={};n=len(batch['state']);valid=int(batch['action_mask'].sum())
        for name in ARMS:
            obs,target,mask=move_to(batch,devices[name]);pred=models[name](**obs)
            if pred.shape!=target.shape or not torch.isfinite(pred).all():raise RuntimeError('Prediction contract')
            err=(pred-target).square();s=totals[name];s['first']+=float(err[:,0].double().sum());s['chunk']+=float(torch.where(mask[...,None],err,0.).double().sum());s['constant']+=float(target[:,0].double().square().sum())
            for task,e in zip(batch['task_id'].tolist(),err[:,0].double().mean(-1).cpu().numpy()):s['per'][task][0]+=float(e);s['per'][task][1]+=1
            if export:predictions[name][frames:frames+n]=pred.cpu().numpy()
        frames+=n;targets+=valid
        if frames%4096==0:write(run/'progress.json',dict(stage=stage,frames=frames,total_frames=len(data)))
    assert frames==30465 and targets==296235
    metrics={name:dict(frames=frames,valid_target_steps=targets,normalized_mse=s['first']/(frames*7),all_valid_chunk_mse=s['chunk']/(targets*7),mean_action_normalized_mse=s['constant']/(frames*7),per_task={str(k):dict(frames=n,mse=e/n) for k,(e,n) in s['per'].items() if n}) for name,s in totals.items()}
    if export:
        for name in ARMS:
            path=run/name/(stage+'_predictions.npy');temp=path.with_suffix('.npy.tmp')
            with temp.open('wb') as f:np.save(f,predictions[name]);f.flush();os.fsync(f.fileno())
            temp.replace(path)
    return metrics

def probe(p,run,examples):
    batches=[collate_cached([examples[(i*17+j)%120] for j in range(128)]) for i in range(4)]
    stats={};states={}
    for mode,devices in [('single_gpu',dict(depth1='cuda:0',depth2='cuda:0')),('dual_gpu',dict(depth1='cuda:0',depth2='cuda:1'))]:
        models=initial_models(p);models={n:m.to(devices[n]).train() for n,m in models.items()};opts={n:optimizer(m) for n,m in models.items()};timings=[]
        for device in sorted(set(devices.values())):torch.cuda.reset_peak_memory_stats(device)
        for i in range(16):
            synchronize(devices.values());tick=time.monotonic();optimize_batch(models,opts,devices,batches[i%4]);synchronize(devices.values());timings.append(time.monotonic()-tick)
        states[mode]={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in models.items()}
        for m in models.values():m.eval()
        val_times=[]
        with torch.no_grad():
            for i in range(8):
                synchronize(devices.values());tick=time.monotonic()
                for n in ARMS:
                    obs,_,_=move_to(batches[i%4],devices[n]);pred=models[n](**obs);assert pred.shape==(128,10,7) and torch.isfinite(pred).all()
                synchronize(devices.values());val_times.append(time.monotonic()-tick)
        stats[mode]=dict(train_pair_batch_seconds=float(np.mean(timings[4:])),validation_pair_batch_seconds=float(np.mean(val_times[2:])),raw_train_seconds=timings,raw_validation_seconds=val_times,peak_allocated_bytes={device:torch.cuda.max_memory_allocated(device) for device in sorted(set(devices.values()))})
        del models,opts;gc.collect()
        for dev in sorted(set(devices.values())):
            with torch.cuda.device(dev):torch.cuda.empty_cache()
    maxerr=0.
    for n in ARMS:
        for k,value in states['single_gpu'][n].items():
            other=states['dual_gpu'][n][k];maxerr=max(maxerr,float((value-other).abs().max()));torch.testing.assert_close(value,other,atol=1e-6,rtol=0)
    speedup=stats['single_gpu']['train_pair_batch_seconds']/stats['dual_gpu']['train_pair_batch_seconds']
    selected='dual_gpu' if speedup>=1.05 else 'single_gpu';s=stats[selected]
    result=dict(status='PASS',samples=120,discarded_updates_per_arm_per_mode=16,parameter_max_error=maxerr,atol=1e-6,modes=stats,measured_training_speedup=speedup,selected_mode=selected,reason='Measured independent-GPU benefit' if selected=='dual_gpu' else 'TwoGPU scheduling offered less than5percent measured benefit; keep same matched batches on oneGPU',estimated_pair50epoch_seconds=1.25*(94950*s['train_pair_batch_seconds']+239*54*s['validation_pair_batch_seconds']))
    write(run/'head_probe.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();pp=Path(a.plan).resolve();p=json.loads(pp.read_text());run=Path(a.run).resolve();started=time.monotonic();torch.set_num_threads(2)
    assert 2<=torch.cuda.device_count()<=4 and torch.cuda.current_device()==0
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
    assert p['training']['epochs']==50 and p['training']['total_updates_per_arm']==94950
    ready=admission(p['resources']);write(run/'memory_admission.json',ready);assert ready['status']=='RAM_READY'
    base=torch.load(R/p['backbone_checkpoint'],map_location='cpu',weights_only=False);front=AdaptedFrontend(R/p['model_directory'],base['adapters']);assert not any(x.requires_grad for x in front.model.parameters())
    write(run/'loading.json',dict(status='PASS',loading=front.loading,frontend_trainable_parameters=0,adapter_count=len(base['adapters']),head_from_regression_endpoint=False));del base
    examples,gate=frontend_gate(p,run,front);profile=probe(p,run,examples);del examples;gc.collect()
    gate_report=json.loads((run/'frontend_gate.json').read_text());estimate=1.5*273465*gate_report['elapsed_seconds']/240+profile['estimated_pair50epoch_seconds']+1800
    write(run/'throughput_admission.json',dict(status='PASS' if estimate<p['resources']['operational_timeout_seconds'] else 'OPERATIONAL_ESTIMATE_BLOCKED',estimated_stage_seconds=estimate,head_pair_seconds=profile['estimated_pair50epoch_seconds'],basis='Measured120 repeated frontend plus paired-head GPU modes;conservative single-frontend extraction projection,not assumed fourGPU multiplier'))
    assert estimate<p['resources']['operational_timeout_seconds'],'Operational budget needs review;no shortened training'
    tokens=extract(p,run,front,gate);del front;gc.collect();torch.cuda.empty_cache()
    data={part:RamTokenData(R/p['cache'],R/p['numeric_cache'],part,tokens) for part in ['train','validation']}
    devices=dict(depth1='cuda:0',depth2='cuda:1' if profile['selected_mode']=='dual_gpu' else 'cuda:0');models={n:m.to(devices[n]) for n,m in initial_models(p).items()};opts={n:optimizer(m) for n,m in models.items()}
    for n in ARMS:(run/n).mkdir(exist_ok=True)
    history={n:[] for n in ARMS};best={n:dict(value=float('inf'),epoch=None,state=None,metrics=None) for n in ARMS};updates=0;pre=json.loads((R/p['preprocessing']).read_text());receipt=sha(run/'cache_receipts.jsonl');science_sha=sha(R/p['scientific_plan']) if p.get('scientific_plan') else sha(pp)
    def snapshot(name,epoch,metrics,state=None):
        return dict(architecture='fixed_adapted_vlm_regression_depth',arm=name,decoder_layers=1 if name=='depth1' else 2,model_config=asdict(models[name].config),state_dict=state if state is not None else {k:v.detach().cpu().clone() for k,v in models[name].state_dict().items()},epoch=epoch,updates=epoch*1899,seed=42,preprocessing=pre,validation=metrics,plan_sha256=sha(pp),scientific_plan_sha256=science_sha,frontend_checkpoint=p['backbone_checkpoint'],frontend_checkpoint_sha256=p['backbone_checkpoint_sha256'],ram_cache_receipts_sha256=receipt,vlm_and_adapters_frozen=True)
    initial=evaluate_pair(models,devices,data['validation'],run,'initial',True)
    for n in ARMS:save(run/n/'initial.pt',snapshot(n,0,initial[n]))
    first_epoch=1
    if p.get('resume'):
        previous=torch.load(R/p['resume']['paired_checkpoint'],map_location='cpu',weights_only=False);assert sha(R/p['resume']['paired_checkpoint'])==p['resume']['sha256'] and previous['scientific_plan_sha256']==science_sha and previous['receipt_sha256']==receipt
        for n in ARMS:models[n].load_state_dict(previous['states'][n]);opts[n].load_state_dict(previous['optimizers'][n])
        history=previous['history'];best=previous['best'];updates=previous['updates'];first_epoch=previous['epoch']+1;assert updates==(first_epoch-1)*1899
        torch.set_rng_state(previous['cpu_rng']);[torch.cuda.set_rng_state(value,device) for device,value in previous['cuda_rng'].items()]
    for epoch in range(first_epoch,51):
        tick=time.monotonic();seen=valid=0;loss_sum={n:0. for n in ARMS};order=hashlib.sha256()
        for m in models.values():m.train()
        for batch in loader(data['train'],shuffle=True,epoch=epoch,workers=0):
            losses=optimize_batch(models,opts,devices,batch);count=int(batch['action_mask'].sum());seen+=len(batch['state']);valid+=count;updates+=1
            order.update(np.stack((batch['episode_id'].numpy(),batch['frame_index'].numpy()),1).astype('<i8').tobytes())
            for n in ARMS:loss_sum[n]+=losses[n]*count*7
            if updates%100==0:
                guard(p['resources']);elapsed=time.monotonic()-tick;write(run/'progress.json',dict(stage='paired_regression_training',epoch=epoch,epochs=50,updates_per_arm=updates,total_updates_per_arm=94950,train_frames_this_epoch=seen,mode=profile['selected_mode'],estimated_training_seconds_remaining=(94950-updates)*elapsed/max(1,(seen+127)//128)))
        assert seen==243000 and valid==2362230 and updates==epoch*1899
        metrics=evaluate_pair(models,devices,data['validation'],run,'epoch_validation');elapsed=time.monotonic()-tick
        for n in ARMS:
            history[n].append(dict(epoch=epoch,updates=updates,train_frames=seen,valid_target_steps=valid,train_mse=loss_sum[n]/(valid*7),sample_order_sha256=order.hexdigest(),validation=metrics[n],paired_epoch_wall_seconds=elapsed))
            if metrics[n]['normalized_mse']<best[n]['value']:best[n]=dict(value=metrics[n]['normalized_mse'],epoch=epoch,state={k:v.detach().cpu().clone() for k,v in models[n].state_dict().items()},metrics=metrics[n])
        # Both arms commit atomically in one bundle, including authoritative best states.
        bundle=dict(epoch=epoch,updates=updates,states={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in models.items()},optimizers={n:o.state_dict() for n,o in opts.items()},history=history,best=best,scientific_plan_sha256=science_sha,plan_sha256=sha(pp),receipt_sha256=receipt,frontend_checkpoint_sha256=p['backbone_checkpoint_sha256'],cpu_rng=torch.get_rng_state(),cuda_rng={device:torch.cuda.get_rng_state(device) for device in sorted(set(devices.values()))},devices=devices)
        save(run/'paired_latest.pt',bundle)
        for n in ARMS:write(run/n/'history.json',history[n])
        print('epoch',epoch,'updates_per_arm',updates,'firstMSE',{n:metrics[n]['normalized_mse'] for n in ARMS},'best',{n:best[n]['epoch'] for n in ARMS},flush=True)
    latest=evaluate_pair(models,devices,data['validation'],run,'latest',True)
    for n in ARMS:save(run/n/'latest.pt',snapshot(n,50,latest[n]));models[n].load_state_dict(best[n]['state'])
    selected=evaluate_pair(models,devices,data['validation'],run,'best',True)
    for n in ARMS:save(run/n/'best.pt',snapshot(n,best[n]['epoch'],selected[n]))
    receipts=[json.loads(x) for x in (run/'cache_receipts.jsonl').read_text().splitlines()]
    for e in receipts:assert token_sha(tokens[e['episode_id']])==e['feature_sha256']
    report=dict(status='COMPLETE_REVIEW_REQUIRED',epochs_per_arm=50,updates_per_arm=94950,arms={n:dict(parameters=sum(v.numel() for v in models[n].parameters()),best_epoch=best[n]['epoch'],selected_validation=selected[n],initial_validation=initial[n],checkpoints={k:sha(run/n/(k+'.pt')) for k in ['initial','best','latest']},prediction_hashes={k:sha(run/n/(k+'_predictions.npy')) for k in ['initial','best','latest']}) for n in ARMS},paired_latest_sha256=sha(run/'paired_latest.pt'),ram_cache_immutable_after_training=True,frontend_trainable_parameters=0,selected_mode=profile['selected_mode'],devices=devices,new_simulation_episodes=0,final2000_accessed=False,total_elapsed_seconds=time.monotonic()-started)
    write(run/'report.json',report);write(run/'progress.json',dict(stage='complete',epochs_per_arm=50,updates_per_arm=94950));return 0
if __name__=='__main__':raise SystemExit(main())
