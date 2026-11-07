"""M3 visual controls; M1/M2 implementations remain immutable."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

from .data import sha256, episode_path, encode_instruction, decode_image
from .model import masked_mse
from .chunk_model_v1 import ChunkCachedData
from .visual_model_v1 import VisualMiniVLA as MiniVLA, VisualConfig as ModelConfig
from .policy import MiniVLAPolicy

ROOT = Path(__file__).resolve().parents[1]

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def write(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)

def checkpoint(path, data):
    tmp=path.with_suffix('.tmp'); torch.save(data,tmp); tmp.replace(path)

def prepare(plan, run):
    start=time.monotonic(); root=ROOT/plan['dataset_root']; cache=ROOT/plan['cache']
    pre=json.loads((ROOT/plan['preprocessing']).read_text()); split=json.loads((ROOT/plan['split']).read_text())
    info=json.loads((root/'meta/info.json').read_text())
    bound={'split_sha256':sha256(ROOT/plan['split']),'preprocessing_sha256':sha256(ROOT/plan['preprocessing']), 'image_size':128}
    complete=cache/'complete.json'
    if complete.exists():
        old=json.loads(complete.read_text()); assert old['binding']==bound
        for part,spec in old['parts'].items():
            for key,shape in spec['shapes'].items():
                assert list(np.load(cache/part/(key+'.npy'),mmap_mode='r').shape)==shape
        write(run/'cache_reference.json',old); return
    cache.mkdir(parents=True,exist_ok=True); all_parts={}; records=[]
    for part in ['train','validation']:
        target=cache/part; target.mkdir(exist_ok=True)
        n=sum(e['length'] for e in split[part])
        specs={'images':((n,2,3,128,128),'uint8'),'state':((n,8),'float32'),
               'actions':((n,1,7),'float32'),'token_ids':((n,32),'int64'),
               'language_mask':((n,32),'bool'),'episode_id':((n,),'int64'),'task_id':((n,),'int64')}
        # Incomplete cache rebuild is safe: it has no model results and no complete manifest.
        maps={k:np.lib.format.open_memmap(target/(k+'.npy'),mode='w+',dtype=dtype,shape=shape) for k,(shape,dtype) in specs.items()}
        offset=0; checks=[]
        for number,e in enumerate(split[part]):
            table=pq.read_table(episode_path(root,info,e['episode_id']),columns=['image','wrist_image','state','actions','episode_index','frame_index'],use_threads=False)
            rows=table.to_pylist(); assert len(rows)==e['length']
            ids,mask=encode_instruction(e['instruction'],pre['vocabulary'])
            for local,row in enumerate(rows):
                assert row['episode_index']==e['episode_id'] and row['frame_index']==local
                i=offset+local
                for camera,key in enumerate(['image','wrist_image']):
                    with Image.open(io.BytesIO(row[key]['bytes'])) as im:
                        rgb=np.asarray(im.convert('RGB').resize((128,128),Image.Resampling.BILINEAR))
                        maps['images'][i,camera]=rgb.transpose(2,0,1)
                    if local==len(rows)//2:
                        expected=decode_image(row[key]).numpy()
                        np.testing.assert_array_equal(maps['images'][i,camera].astype('float32')/255,expected)
                for key in ['state','actions']:
                    stats=pre['normalization'][key]
                    value=(np.asarray(row[key],dtype='float32')-np.asarray(stats['mean'],dtype='float32'))/np.asarray(stats['scale'],dtype='float32')
                    maps[key][i]=value if key=='state' else value[None]
                maps['token_ids'][i]=ids.numpy(); maps['language_mask'][i]=mask.numpy()
                maps['episode_id'][i]=e['episode_id'];maps['task_id'][i]=e['task_id']
            checks.append(e['episode_id']);offset+=len(rows)
            if number%10==0 or offset==n:
                elapsed=time.monotonic()-start
                progress={'stage':'cache','partition':part,'episodes':number+1,'total_episodes':len(split[part]),'frames':offset,'total_frames':n,'elapsed_seconds':elapsed,'updated_at':now()}
                write(run/'progress.json',progress); print(json.dumps(progress),flush=True)
        assert offset==n
        for mmap in maps.values():mmap.flush()
        del maps
        all_parts[part]={'frames':n,'episodes':len(split[part]),'shapes':{k:list(v[0]) for k,v in specs.items()},'camera_exact_checks':len(checks)*2}
    record={'status':'PASS','binding':bound,'parts':all_parts,'elapsed_seconds':time.monotonic()-start,'created_at':now(),
            'bytes':sum(p.stat().st_size for p in cache.glob('*/*.npy'))}
    assert record['bytes']<6*2**30
    write(complete,record);write(run/'cache_reference.json',record)

class CachedData(Dataset):
    def __init__(self, cache, part):
        self.directory=Path(cache)/part
        self.arrays={p.stem:np.load(p,mmap_mode='r') for p in self.directory.glob('*.npy')}
    def __len__(self):return len(self.arrays['state'])
    def __getitem__(self,i):return {k:torch.from_numpy(np.array(v[i],copy=True)) for k,v in self.arrays.items()}

def inputs(batch,device):
    im=batch['images'].to(device,non_blocking=True).float()/255
    return dict(image=im[:,0],wrist_image=im[:,1],state=batch['state'].to(device,non_blocking=True),
                token_ids=batch['token_ids'].to(device,non_blocking=True),language_mask=batch['language_mask'].to(device,non_blocking=True))

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); squared=0.; frames=0; per_dim=np.zeros(7); per_task={}; constant=0.; chunk_sse=0.; chunk_values=0
    for batch in loader:
        target=batch['actions'].to(device,non_blocking=True)
        prediction=model(**inputs(batch,device)); errors=(prediction[:,:1]-target[:,:1]).square()
        mask=batch['action_mask'].to(device).unsqueeze(-1)
        chunk_sse+=float(((prediction-target).square()*mask).sum());chunk_values+=int(mask.sum())*7
        squared+=float(errors.sum());constant+=float(target[:,:1].square().sum());frames+=len(target)
        per_dim+=errors.sum(dim=(0,1)).cpu().numpy()
        values=errors.mean(dim=(1,2)).cpu().numpy()
        for task,value in zip(batch['task_id'].numpy(),values):
            pair=per_task.setdefault(str(int(task)),[0.,0]);pair[0]+=float(value);pair[1]+=1
    return {'frames':frames,'normalized_mse':squared/(frames*7),'all_valid_chunk_mse':chunk_sse/chunk_values,'valid_target_steps':chunk_values//7,'mean_action_normalized_mse':constant/(frames*7),
            'per_dimension_normalized_mse':(per_dim/frames).tolist(),
            'per_task':{k:{'mse':v[0]/v[1],'frames':v[1]} for k,v in per_task.items()}}

def train(plan,run):
    t=plan['training'];torch.set_num_threads(plan['resources']['max_cpu_threads'])
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    device=torch.device('cuda:0');torch.manual_seed(t['seed']);np.random.seed(t['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    pre=json.loads((ROOT/plan['preprocessing']).read_text())
    model=MiniVLA(ModelConfig(len(pre['vocabulary']),horizon=10,vision_variant=plan['model_variant']),initialize_pretrained=True,pretrained_path=ROOT/plan['pretraining']['path']).to(device)
    assert sum(p.numel() for p in model.parameters())==plan['expected_parameters'][plan['model_variant']]
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    train_data=ChunkCachedData(ROOT/plan['cache'],'train');val_data=ChunkCachedData(ROOT/plan['cache'],'validation')
    assert len(train_data)==t['expected_train_frames_per_epoch'] and len(val_data)==t['expected_validation_frames']
    val_loader=DataLoader(val_data,batch_size=t['batch_size'],shuffle=False,num_workers=t['workers'],pin_memory=True,persistent_workers=True)
    generator=torch.Generator()
    loader=DataLoader(train_data,batch_size=t['batch_size'],sampler=RandomSampler(train_data,generator=generator),
                      generator=torch.Generator().manual_seed(t['seed']+10000),num_workers=t['workers'],pin_memory=True,persistent_workers=True,drop_last=False)
    directory=run/'training';directory.mkdir(exist_ok=True)
    base={'model_variant':plan['model_variant'],'pretraining_sha256':sha256(ROOT/plan['pretraining']['path']) if plan['model_variant']=='resnet_pretrained' else None,'model_config':model.configuration(),'preprocessing':pre,'seed':t['seed'],'plan_sha256':sha256(Path(os.environ['MINI_VLA_PLAN']))}
    best=float('inf');first_epoch=1;updates=0;history=[];prior_seconds=0.
    if (directory/'latest.pt').exists():
        ck=torch.load(directory/'latest.pt',map_location=device,weights_only=False)
        assert ck['plan_sha256']==base['plan_sha256']
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        first_epoch=ck['epoch']+1;updates=ck['updates'];best=ck['best_mse'];history=ck['history'];prior_seconds=ck['training_elapsed_seconds']
    else:
        assert not (directory/'initial.pt').exists(),'Partial initialization requires recorded recovery'
        checkpoint(directory/'initial.pt',{**base,'state_dict':model.state_dict(),'epoch':0,'updates':0})
        initial=evaluate(model,val_loader,device);write(directory/'initial_validation.json',initial)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch,t['epochs']+1):
        generator.manual_seed(t['seed']+epoch);model.train();acc=0.;seen=0;target_count=0;epoch_start=time.monotonic()
        for number,batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            target=batch['actions'].to(device,non_blocking=True)
            pred=model(**inputs(batch,device));loss=masked_mse(pred,target,batch['action_mask'].to(device))
            assert torch.isfinite(loss)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm']);assert torch.isfinite(gn)
            optimizer.step();updates+=1;seen+=len(target);valid=int(batch['action_mask'].sum());target_count+=valid;acc+=float(loss.detach())*valid
            if number%50==0:
                elapsed=time.monotonic()-start; finished=(epoch-first_epoch)*len(train_data)+seen
                remaining=(t['epochs']-epoch)*len(train_data)+len(train_data)-seen
                progress={'stage':'training','epoch':epoch,'epochs':t['epochs'],'frames_in_epoch':seen,'train_frames':len(train_data),'updates':updates,
                          'training_mse_running':acc/target_count,'elapsed_seconds':prior_seconds+elapsed,'estimated_remaining_training_seconds':elapsed*remaining/max(1,finished),'updated_at':now()}
                write(run/'progress.json',progress);print(json.dumps(progress),flush=True)
        assert seen==len(train_data)
        assert target_count==t['valid_target_steps_per_epoch']
        val=evaluate(model,val_loader,device)
        assert val['valid_target_steps']==t['expected_validation_target_steps']
        row={'epoch':epoch,'updates':updates,'train_frames':seen,'valid_target_steps':target_count,'train_mse':acc/target_count,'validation':val,'epoch_seconds':time.monotonic()-epoch_start}
        history.append(row);improved=val['normalized_mse']<best
        best=min(best,val['normalized_mse'])
        record={**base,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'updates':updates,'best_mse':best,
                'history':history,'training_elapsed_seconds':prior_seconds+time.monotonic()-start}
        checkpoint(directory/'latest.pt',record)
        if improved:checkpoint(directory/'best.pt',record)
        write(directory/'history.json',history);print(json.dumps(row),flush=True)
    best_ck=torch.load(directory/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_ck['state_dict'])
    final=evaluate(model,val_loader,device)
    assert abs(final['normalized_mse']-best_ck['best_mse'])<1e-6
    # Warm steady-state batch-one compute latency; excludes image preprocessing/RPC.
    one={k:v[:1] for k,v in next(iter(val_loader)).items()};x=inputs(one,device)
    with torch.no_grad():
        for _ in range(10):model(**x)
        torch.cuda.synchronize();tick=time.monotonic()
        for _ in range(100):model(**x)
        torch.cuda.synchronize();latency=(time.monotonic()-tick)/100
    report={'status':'PASS','model_variant':plan['model_variant'],'parameters':sum(p.numel() for p in model.parameters()),'pretraining_sha256':base['pretraining_sha256'],'epochs':t['epochs'],'updates':updates,'training_seed':t['seed'],'best_epoch':best_ck['epoch'],'selected_validation':final,
            'training_elapsed_seconds':prior_seconds+time.monotonic()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'steady_batch1_forward_seconds':latency,
            'checkpoint_sha256':{k:sha256(directory/(k+'.pt')) for k in ['initial','best','latest']},'completed_at':now()}
    write(directory/'report.json',report)

def serve(checkpoint_file,port):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    ck=torch.load(checkpoint_file,map_location='cpu',weights_only=False)
    model=MiniVLA(ModelConfig(**ck['model_config'])).cuda();model.load_state_dict(ck['state_dict']);model.eval()
    policy=MiniVLAPolicy(model,ck['preprocessing']['vocabulary'],ck['preprocessing']['normalization'])
    WebsocketPolicyServer(policy,host='127.0.0.1',port=port,metadata={'model':'mini_vla_m3','epoch':ck['epoch'],'action_horizon':10,'checkpoint_sha256':sha256(Path(checkpoint_file))}).serve_forever()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','train','serve']);p.add_argument('--plan',required=True);p.add_argument('--run',required=True);p.add_argument('--variant');p.add_argument('--checkpoint');p.add_argument('--port',type=int);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run);os.environ['MINI_VLA_PLAN']=str(Path(a.plan).resolve())
    plan['model_variant']=a.variant
    if a.mode=='cache':prepare(plan,run)
    elif a.mode=='train':train(plan,run)
    else:serve(a.checkpoint,a.port)
if __name__=='__main__':main()
