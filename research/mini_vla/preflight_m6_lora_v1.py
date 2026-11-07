"""Bounded real-input gradient/identity/throughput probe. No retained trained model."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from .smolvlm_lora_v1 import DifferentiableFrontend,adapter_parameters
from .vlm_frontend_v1 import decode_record,prompt_for
from .vlm_action_heads_v1 import RegressionHead,HeadConfig
from .vlm_cached_data_v2 import VLMCacheData
from .model import masked_mse

ROOT=Path(__file__).resolve().parents[1]
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);args=ap.parse_args()
    plan=json.loads(Path(args.plan).read_text());run=Path(args.run);started=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.device_count()==1
    for rel,h in plan['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    downloads=json.loads((ROOT/plan['model_manifest']).read_text())
    for name,meta in downloads['files'].items():
        f=ROOT/plan['model_directory']/name
        assert f.stat().st_size==meta['bytes'] and sha(f)==meta['sha256'],name
    split=json.loads((ROOT/plan['split']).read_text());allowed={x['episode_id'] for x in split['train']};forbidden={x['episode_id'] for x in split['validation']}
    samples=json.loads((ROOT/plan['sample_manifest']).read_text())['samples'];assert len(samples)==120
    assert {s['episode_id'] for s in samples}.issubset(allowed-forbidden)
    recorded_gate=json.loads((ROOT/plan['cache']/'gate/complete.json').read_text())
    data=VLMCacheData(ROOT/plan['cache'],ROOT/plan['numeric_cache'],'train',max_open_episodes=2)
    index={e['episode_id']:e for e in data.index}
    ck=torch.load(ROOT/plan['head_checkpoint'],map_location='cpu',weights_only=False)
    assert ck['epoch']==12 and ck['architecture']=='frozen_vlm_regression'
    head=RegressionHead(HeadConfig(**ck['model_config'])).cuda().eval();head.load_state_dict(ck['state_dict'])
    # Original unwrapped model gives the online zero-adapter identity reference.
    from .vlm_frontend_v1 import FrozenFrontend
    base=FrozenFrontend(str(ROOT/plan['model_directory']))
    records=[];nums=[];references=[];io_times=[];input_hashes=[]
    for i,s in enumerate(samples):
        t0=time.perf_counter();eid=s['episode_id'];ep=index[eid];frame=s['frame_index']
        file=ROOT/plan['dataset_root']/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
        row=pq.read_table(file,columns=['image','wrist_image','episode_index','frame_index'],use_threads=False).slice(frame,1).to_pylist()[0]
        assert row['episode_index']==eid and row['frame_index']==frame
        record=decode_record(row,s['instruction']);num=data[ep['start']+frame];assert num['episode_id']==eid and num['frame_index']==frame
        online=base.encode([record])[0]
        # Source identity must match the already-approved cache input, not merely its shape.
        gate_row=recorded_gate['comparisons'][i]
        assert gate_row['episode_id']==eid and gate_row['frame_index']==frame
        assert gate_row['input_sha256']==online['input_sha256']
        # Old cache remains a fixed reference only; never used by the adapting arm.
        ref=num['hidden'].float();live=torch.from_numpy(online['bits']).view(torch.bfloat16).float()
        rel=float(torch.linalg.vector_norm(live-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
        assert rel<=1e-2,('online/cache original threshold',i,rel)
        with torch.no_grad():pred=head(hidden=live[None].cuda(),hidden_mask=torch.ones(1,len(live),dtype=torch.bool,device='cuda'),state=num['state'][None].cuda()).cpu()
        references.append(dict(bits=online['bits'],pred=pred,cache_relative_L2=rel))
        records.append(record);nums.append(num);input_hashes.append(dict(episode_id=eid,frame_index=frame,sha256=online['input_sha256']))
        io_times.append(time.perf_counter()-t0)
        if i%10==0:write(run/'progress.json',dict(stage='baseline_120_inputs',done=i+1,total=120))
    del base;torch.cuda.empty_cache()
    torch.manual_seed(42)
    front=DifferentiableFrontend(str(ROOT/plan['model_directory']),rank=8,alpha=8,checkpointing=True).cuda().eval()
    parity=[]
    for i,(record,num,ref) in enumerate(zip(records,nums,references)):
        inputs=front.inputs([record])
        with torch.no_grad():
            hidden,mask=front(inputs);pred=head(hidden=hidden,hidden_mask=mask,state=num['state'][None].cuda())
        bits=hidden[0,mask[0]].contiguous().cpu().view(torch.uint16).numpy()
        assert np.array_equal(bits,ref['bits']),('zero_adapter_hidden_identity',i)
        error=float((pred.cpu()-ref['pred']).abs().max());assert error<=1e-6
        parity.append(error)
    write(run/'parity.json',dict(status='PASS',samples=120,input_hashes=input_hashes,zero_adapter_hidden_bits_exact=True,max_action_error=max(parity),cache_relative_L2_max=max(r['cache_relative_L2'] for r in references)))
    initial_adapters={n:p.detach().clone() for n,p in adapter_parameters(front.model).items()}
    # Frozen and trainable probes start from identical head/adapter weights; disposable updates only.
    measurements={}
    for mode in ['frozen','lora']:
        head.load_state_dict(ck['state_dict']);head.train()
        for n,p in adapter_parameters(front.model).items():p.data.copy_(initial_adapters[n])
        front.enable_adapter_training(mode=='lora');front.train(mode=='lora')
        opt=torch.optim.AdamW([{'params':head.parameters(),'lr':3e-5},
            {'params':list(adapter_parameters(front.model).values()) if mode=='lora' else [],'lr':1e-5}],weight_decay=1e-4)
        raw=[];gradient_groups={};updated_groups=set();torch.cuda.reset_peak_memory_stats()
        for step in range(12):
            torch.cuda.synchronize();tic=time.perf_counter();opt.zero_grad(set_to_none=True)
            # Four fixed training observations per disposable optimizer step, microbatch1.
            ids=[(step*4+j)%len(records) for j in range(4)]
            denom=7*sum(int(nums[i]['action_mask'].sum()) for i in ids)
            for i in ids:
                inputs=front.inputs([records[i]])
                with torch.set_grad_enabled(mode=='lora'):hidden,mask=front(inputs)
                pred=head(hidden=hidden,hidden_mask=mask,state=nums[i]['state'][None].cuda())
                target=nums[i]['actions'][None].cuda();valid=nums[i]['action_mask'][None,:,None].cuda()
                loss=torch.where(valid,(pred-target).square(),0.).sum()/denom
                assert torch.isfinite(loss);loss.backward()
            for name,p in front.model.named_parameters():
                if not name.endswith(('.lora_a','.lora_b')):assert p.grad is None,name
                elif mode=='lora' and p.grad is not None:
                    assert torch.isfinite(p.grad).all(),name
                    group='vision' if 'vision_model' in name else 'text'
                    gradient_groups[group]=gradient_groups.get(group,0)+int(bool(p.grad.abs().max()>0))
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize();raw.append(time.perf_counter()-tic)
            write(run/'progress.json',dict(stage=mode+'_disposable_backward_probe',done=step+1,total=12))
        if mode=='lora':
            for n,p in adapter_parameters(front.model).items():
                if not torch.equal(p,initial_adapters[n]):updated_groups.add('vision' if 'vision_model' in n else 'text')
            assert set(gradient_groups)==updated_groups=={'vision','text'}
        measurements[mode]=dict(seconds_per_example=float(np.mean(raw[3:]))/4,raw_step_seconds=raw,microbatch=1,disposable_updates=12,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gradient_groups=gradient_groups,updated_groups=sorted(updated_groups))
        del opt
    front.eval();head.eval();forward=[]
    with torch.no_grad():
        for record in records[:24]:
            torch.cuda.synchronize();t0=time.perf_counter();front(front.inputs([record]));torch.cuda.synchronize();forward.append(time.perf_counter()-t0)
    fwd=float(np.mean(forward[3:]));train_per=measurements['lora']['seconds_per_example']
    # No claimed two-GPU speedup. Estimates omit full-source random I/O: explicit margin and boundary.
    estimated=1.25*(243000*train_per+2*30465*fwd+900+1200)
    result=dict(status='COMPLETE_REVIEW_REQUIRED',scope='Engineering probe only; no retained model or new simulation',measurements=measurements,
                validation_forward_seconds_per_example=fwd,estimated_one_gpu_lora_one_epoch_plus_two_validations_and_control_eval_seconds=estimated,
                estimate_boundary='Source RGB decode/randomIO and twoGPU scaling not fully measured. Frozen control may use verified old cache. 25% margin and 2100s control/eval/audit allowance are assumptions.',
                full_50epoch_lora_training_seconds_without_validation=243000*50*train_per,
                examples=120,retained_training_updates=0,discarded_optimizer_updates=24,new_simulation_episodes=0,
                elapsed_seconds=time.monotonic()-started,trainable_lora_parameters=2228224,trainable_head_parameters=398663)
    write(run/'result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
