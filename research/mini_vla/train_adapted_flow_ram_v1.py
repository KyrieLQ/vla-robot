"""Pinned adapted VLM -> full lossless RAM tokens -> newly initialized Flow head.
Only head training; no final2000 access. Cache metadata/small gate and head checkpoints persist.
"""
import argparse,gc,hashlib,json,os,shutil,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import pyarrow.parquet as pq
from .policy_paired_v1 import AdaptedFrontend
from .paired_continuation_v1 import RawData
from .vlm_frontend_v1 import decode_record,bits_bf16
from .vlm_cached_data_v2 import collate_cached
from .ram_token_data_v1 import RamTokenData
from .vlm_action_heads_v1 import FlowHead,HeadConfig
from .train_vlm_flow_v1 import seed,optimizer,loader,move,training_loss,fixed_noise,evaluate
from .train_vlm_head_v1 import sha,write,save
from flow_ram_resources_v1 import admission,guard,current_rss,host_available
R=Path(__file__).resolve().parents[1]

def token_sha(a):return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

@torch.no_grad()
def frontend_gate(p,run,frontend):
    started=time.monotonic();directory=run/'gate';directory.mkdir(exist_ok=True)
    raw=RawData(p,'train');index={e['episode_id']:e for e in raw.index};samples=json.loads((R/p['sample_manifest']).read_text())['samples'];old=json.loads((R/p['cache']/'gate/complete.json').read_text())['comparisons'];examples=[];rows=[]
    for i,s in enumerate(samples):
        e=index[s['episode_id']];num=raw[e['start']+s['frame_index']];record=num.pop('record')
        a=frontend.encode([record])[0];b=frontend.encode([record])[0]
        assert np.array_equal(a['bits'],b['bits']) and a['input_sha256']==old[i]['input_sha256']
        assert a['bits'].shape==(e['tokens'],960)
        path=directory/f'input_{i:03d}.npy';np.save(path,a['bits']);assert np.array_equal(np.load(path),a['bits'])
        num.update(hidden=bits_bf16(a['bits']),hidden_mask=torch.ones(e['tokens'],dtype=torch.bool));examples.append(num)
        rows.append(dict(episode_id=s['episode_id'],frame_index=s['frame_index'],input_sha256=a['input_sha256'],feature_sha256=token_sha(a['bits']),file=path.name,file_sha256=sha(path)))
        if i%10==0:write(run/'progress.json',dict(stage='adapted_frontend_gate',done=i+1,total=120))
    assert len(rows)==120
    result=dict(status='PASS',samples=120,rows=rows,repeat_features_bitwise_equal=True,stored_gate_roundtrip_exact=True,reference='same selected adapted VLM batch1; not old frozen hidden features',elapsed_seconds=time.monotonic()-started)
    write(run/'frontend_gate.json',result);return examples,rows

def probe(p,run,examples):
    seed(42);model=FlowHead(HeadConfig(**p['head_config'])).cuda();opt=optimizer(model);steps=[];validation=[]
    batches=[collate_cached([examples[(i*17+j)%120] for j in range(128)]) for i in range(4)]
    torch.cuda.reset_peak_memory_stats()
    for i in range(12):
        obs,target,mask=move(batches[i%4]);torch.cuda.synchronize();tick=time.monotonic();opt.zero_grad(set_to_none=True);loss=training_loss(model,obs,target,mask);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();steps.append(time.monotonic()-tick)
    model.eval()
    with torch.no_grad():
        for i in range(6):
            b=batches[i%4];obs,_,_=move(b);torch.cuda.synchronize();tick=time.monotonic();out=model.sample(**obs,noise=fixed_noise(b).cuda());assert out.shape==(128,10,7) and torch.isfinite(out).all();torch.cuda.synchronize();validation.append(time.monotonic()-tick)
    result=dict(status='PASS',discarded_head_updates=12,effective_batch=128,train_batch_seconds=float(np.mean(steps[3:])),validation_batch_seconds=float(np.mean(validation[2:])),raw_train_seconds=steps,raw_validation_seconds=validation,peak_allocated_bytes=torch.cuda.max_memory_allocated(),estimate_head50epoch_seconds=1.25*(94950*np.mean(steps[3:])+239*51*np.mean(validation[2:])),scope='fixed120 original train inputs reused only for discarded shape/gradient/throughput probe;RAM full iteration overhead not included')
    write(run/'head_probe.json',result);del model,opt,batches;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def extract(p,run,front,gate_rows):
    started=time.monotonic();tokens={};receipts=[];count=0;total_bytes=0;split=json.loads((R/p['split']).read_text());metas={e['episode_id']:e for part in ['train','validation'] for e in split[part]}
    assert len(split['train'])==1506 and len(split['validation'])==187
    assert not {e['episode_id'] for e in split['train']}&{e['episode_id'] for e in split['validation']}
    source=json.loads((R/p['source_integrity']).read_text());source_hash={x['path']:x['sha256'] for x in source['files']};gate={(x['episode_id'],x['frame_index']):x for x in gate_rows};gate_seen=0
    for part in ['train','validation']:
        layout=json.loads((R/p['cache']/part/'index.json').read_text())
        for e in layout:
            guard(p['resources']);eid=e['episode_id'];meta=metas[eid];n=e['end']-e['start'];assert n==meta['length']
            relative=f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet';path=R/p['dataset_root']/relative;assert sha(path)==source_hash[relative]
            rows=pq.read_table(path,columns=['image','wrist_image','episode_index','frame_index'],use_threads=False).to_pylist();assert len(rows)==n
            values=np.empty((n,e['tokens'],960),dtype=np.uint16)
            hashes=json.loads((R/p['cache']/e['directory']/'input_hashes.json').read_text());digest=hashlib.sha256()
            for j,row in enumerate(rows):
                assert row['episode_index']==eid and row['frame_index']==j
                feature=front.encode([decode_record(row,meta['instruction'])])[0];bits=feature['bits'];assert bits.shape==(e['tokens'],960)
                old=hashes[j]['input_sha256'] if isinstance(hashes[j],dict) else hashes[j];assert feature['input_sha256']==old
                values[j]=bits;digest.update(feature['input_sha256'].encode())
                if (eid,j) in gate:
                    expected=gate[eid,j];assert token_sha(bits)==expected['feature_sha256'];gate_seen+=1
                count+=1
                if count%100==0:
                    guard(p['resources']);elapsed=time.monotonic()-started;write(run/'progress.json',dict(stage='full_adapted_RAM_extraction',frames=count,total_frames=273465,episode_id=eid,ram_feature_bytes=total_bytes+values.nbytes,rss_bytes=current_rss(),host_available_bytes=host_available(),elapsed_seconds=elapsed,estimated_remaining_seconds=(273465-count)*elapsed/count))
            assert np.isfinite(bits_bf16(values).float().numpy()).all()
            values.flags.writeable=False;tokens[eid]=values;total_bytes+=values.nbytes
            receipt=dict(part=part,episode_id=eid,frames=n,shape=list(values.shape),dtype='uint16 rawBF16',feature_sha256=token_sha(values),source_sha256=source_hash[relative],input_sequence_sha256=digest.hexdigest())
            receipts.append(receipt)
            with (run/'cache_receipts.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n');f.flush()
    assert count==273465 and len(tokens)==1693 and total_bytes==p['resources']['ram_cache_bytes'] and gate_seen==120
    # Full second pass checks all retained RAM bytes against the committed episode receipts.
    for e in receipts:assert token_sha(tokens[e['episode_id']])==e['feature_sha256']
    complete=dict(status='PASS',frames=count,episodes=len(tokens),bytes=total_bytes,all_source_hashes_verified=True,all_original_input_hashes_verified=True,allRAM_receipts_second_pass_verified=True,fixed120_gate_matches=True,elapsed_seconds=time.monotonic()-started,cache_storage='processRAM_only;not persistent;old disk cache untouched',frontend_checkpoint_sha256=p['backbone_checkpoint_sha256'],receipt_sha256=sha(run/'cache_receipts.jsonl'))
    write(run/'ram_cache_acceptance.json',complete);return tokens

@torch.no_grad()
def evaluate_saved(model,data,prediction_path,run,stage):
    model.eval();preds=np.empty((len(data),10,7),np.float32);offset=0
    for batch in loader(data,workers=0):
        obs,target,mask=move(batch);prediction=model.sample(**obs,noise=fixed_noise(batch).cuda());assert torch.isfinite(prediction).all();n=len(prediction);preds[offset:offset+n]=prediction.cpu().numpy();offset+=n
        if offset%4096==0:write(run/'progress.json',dict(stage=stage,frames=offset,total_frames=len(data)))
    assert offset==30465
    temp=prediction_path.with_suffix('.npy.tmp')
    with temp.open('wb') as f:np.save(f,preds);f.flush();os.fsync(f.fileno())
    temp.replace(prediction_path)
    return sha(prediction_path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();pp=Path(a.plan).resolve();p=json.loads(pp.read_text());run=Path(a.run);started=time.monotonic();torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
    ready=admission(p['resources']);write(run/'memory_admission.json',ready);assert ready['status']=='RAM_READY',ready
    ck=torch.load(R/p['backbone_checkpoint'],map_location='cpu',weights_only=False);assert sha(R/p['backbone_checkpoint'])==p['backbone_checkpoint_sha256'] and ck['arm']=='lora'
    front=AdaptedFrontend(R/p['model_directory'],ck['adapters']);assert not any(x.requires_grad for x in front.model.parameters())
    write(run/'loading.json',dict(status='PASS',loading=front.loading,frontend_trainable_parameters=0,adapter_count=len(ck['adapters']),head_state_from_regression_loaded=False))
    examples,gate=frontend_gate(p,run,front);probe(p,run,examples);del examples;gc.collect()
    tokens=extract(p,run,front,gate);del front;gc.collect();torch.cuda.empty_cache()
    data={part:RamTokenData(R/p['cache'],R/p['numeric_cache'],part,tokens) for part in ['train','validation']}
    seed(42);model=FlowHead(HeadConfig(**p['head_config'])).cuda();opt=optimizer(model);assert sum(x.numel() for x in model.parameters())==432711
    old_initial=torch.load(R/p['old_flow_initial'],map_location='cpu',weights_only=False);assert all(torch.equal(v.cpu(),old_initial['state_dict'][k]) for k,v in model.state_dict().items());del old_initial
    science_sha=sha(R/p['scientific_plan']) if p.get('scientific_plan') else sha(pp)
    out=run/'training';out.mkdir(exist_ok=True);history=[];updates=0;best_value=float('inf');best_epoch=None;pre=json.loads((R/p['preprocessing']).read_text())
    def checkpoint(epoch,metrics):
        return dict(architecture='fixed_adapted_vlm_flow',model_config=asdict(model.config),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch,updates=updates,seed=42,plan_sha256=sha(pp),scientific_plan_sha256=science_sha,frontend_checkpoint=p['backbone_checkpoint'],frontend_checkpoint_sha256=p['backbone_checkpoint_sha256'],ram_cache_receipts_sha256=sha(run/'cache_receipts.jsonl'),preprocessing=pre,history=history,best_epoch=best_epoch,best_value=best_value,validation=metrics,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),vlm_and_adapters_frozen=True)
    initial=evaluate(model,loader(data['validation'],workers=0));write(out/'initial_validation.json',initial);save(out/'initial.pt',checkpoint(0,initial));evaluate_saved(model,data['validation'],out/'initial_predictions.npy',run,'initial_validation_prediction_export')
    start_epoch=1
    if p.get('resume'):
        resume=p['resume'];assert sha(R/resume['checkpoint'])==resume['checkpoint_sha256'] and sha(R/resume['best_checkpoint'])==resume['best_checkpoint_sha256']
        previous=torch.load(R/resume['checkpoint'],map_location='cpu',weights_only=False);assert previous['scientific_plan_sha256']==science_sha and previous['frontend_checkpoint_sha256']==p['backbone_checkpoint_sha256']
        assert previous['ram_cache_receipts_sha256']==sha(run/'cache_receipts.jsonl')
        model.load_state_dict(previous['state_dict']);opt.load_state_dict(previous['optimizer']);updates=previous['updates'];history=previous['history'];best_epoch=previous['best_epoch'];best_value=previous['best_value'];start_epoch=previous['epoch']+1
        assert updates==previous['epoch']*1899 and len(history)==previous['epoch']
        torch.set_rng_state(previous['cpu_rng']);torch.cuda.set_rng_state_all(previous['cuda_rng']);shutil.copyfile(R/resume['best_checkpoint'],out/'best.pt');save(out/'latest.pt',previous)
    for epoch in range(start_epoch,51):
        model.train();seen=targets=0;train_sum=0.;tick=time.monotonic()
        for batch in loader(data['train'],shuffle=True,epoch=epoch,workers=0):
            obs,target,mask=move(batch);opt.zero_grad(set_to_none=True);loss=training_loss(model,obs,target,mask);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();n=int(mask.sum());train_sum+=float(loss.detach())*n*7;targets+=n;seen+=len(target);updates+=1
            if updates%100==0:
                guard(p['resources']);write(run/'progress.json',dict(stage='flow_head_training',epoch=epoch,epochs=50,updates=updates,total_updates=94950,frames_in_epoch=seen,train_velocity_mse=train_sum/(targets*7)))
        assert seen==243000 and targets==2362230 and updates==epoch*1899
        metrics=evaluate(model,loader(data['validation'],workers=0));assert metrics['frames']==30465 and metrics['valid_target_steps']==296235 and len(metrics['per_task'])==40
        history.append(dict(epoch=epoch,updates=updates,train_frames=seen,valid_target_steps=targets,train_velocity_mse=train_sum/(targets*7),validation=metrics,elapsed_seconds=time.monotonic()-tick))
        if metrics['normalized_mse']<best_value:best_value=metrics['normalized_mse'];best_epoch=epoch;save(out/'best.pt',checkpoint(epoch,metrics))
        save(out/'latest.pt',checkpoint(epoch,metrics));write(out/'history.json',history);print('epoch',epoch,'firstMSE',metrics['normalized_mse'],'best',best_epoch,flush=True)
    latest_prediction_hash=evaluate_saved(model,data['validation'],out/'latest_predictions.npy',run,'latest_validation_prediction_export')
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best['state_dict']);best_hash=evaluate_saved(model,data['validation'],out/'best_predictions.npy',run,'best_validation_prediction_export')
    final=evaluate(model,loader(data['validation'],workers=0));assert abs(final['normalized_mse']-best['validation']['normalized_mse'])<1e-9
    receipts=[json.loads(x) for x in (run/'cache_receipts.jsonl').read_text().splitlines()]
    for e in receipts:assert token_sha(tokens[e['episode_id']])==e['feature_sha256']
    write(out/'report.json',dict(status='COMPLETE_REVIEW_REQUIRED',epochs=50,updates=updates,best_epoch=best_epoch,selected_validation=final,initial_validation=initial,frontend_frozen=True,trainable_head_parameters=432711,checkpoints={n:sha(out/(n+'.pt')) for n in ['initial','best','latest']},prediction_hashes={'initial':sha(out/'initial_predictions.npy'),'best':best_hash,'latest':latest_prediction_hash},ram_cache_immutable_after_training=True,total_elapsed_seconds=time.monotonic()-started,new_simulation_episodes=0,final2000_accessed=False))
    write(run/'progress.json',dict(stage='complete',epochs=50,updates=updates,best_epoch=best_epoch));return 0
if __name__=='__main__':raise SystemExit(main())
