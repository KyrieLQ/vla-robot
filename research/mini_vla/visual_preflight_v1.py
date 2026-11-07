"""Bounded official-weight fetch, CPU weight check and synthetic GPU throughput gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
import torch
from .experiment_v1 import ROOT,write
from .model import masked_mse
from .visual_model_v1 import VisualConfig,VisualMiniVLA

def weights(plan,run):
    spec=plan['pretraining'];path=ROOT/spec['path'];path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix('.part')
    start=time.monotonic()
    if not path.exists():
        for attempt in range(3):
            try:
                offset=part.stat().st_size if part.exists() else 0
                request=urllib.request.Request(spec['url'],headers={'Range':'bytes=%d-'%offset} if offset else {})
                with urllib.request.urlopen(request,timeout=30) as response:
                    if offset and response.status==206:
                        assert response.headers['Content-Range'].startswith('bytes %d-'%offset);mode='ab'
                    elif response.status==200:mode='wb';offset=0
                    else:raise RuntimeError('Unexpected weight response')
                    with part.open(mode) as stream:
                        while True:
                            block=response.read(2**20)
                            if not block:break
                            stream.write(block);offset+=len(block)
                            if offset>spec['max_bytes']:raise RuntimeError('Weight size exceeded cap')
                            if offset%(8*2**20)==0:print(json.dumps({'stage':'weights','bytes':offset,'elapsed_seconds':time.monotonic()-start}),flush=True)
                digest=hashlib.sha256(part.read_bytes()).hexdigest();assert digest.startswith(spec['sha256_prefix'])
                part.replace(path);break
            except Exception:
                if attempt==2:raise
                time.sleep(1)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest.startswith(spec['sha256_prefix']) and path.stat().st_size<=spec['max_bytes']
    torch.set_num_threads(4);torch.manual_seed(42)
    model=VisualMiniVLA(VisualConfig(28,horizon=10,vision_variant='resnet_pretrained'),initialize_pretrained=True,pretrained_path=path)
    original=torch.load(path,map_location='cpu',weights_only=True)
    for name,value in [('conv1.weight',model.vision.backbone[0].weight),('bn1.running_mean',model.vision.backbone[1].running_mean)]:
        torch.testing.assert_close(value,original[name],atol=0,rtol=0)
    write(run/'weights.json',{'status':'PASS','path':str(path),'sha256':digest,'bytes':path.stat().st_size,'url':spec['url'],'all_backbone_keys_loaded_strictly':True,'classifier_discarded':True,'elapsed_seconds':time.monotonic()-start})

def profile(plan,run):
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    records={};b=plan['training']['batch_size']
    for variant in ['tiny_norm','resnet_random']:
        torch.manual_seed(42);model=VisualMiniVLA(VisualConfig(28,horizon=10,vision_variant=variant)).cuda().train()
        opt=torch.optim.AdamW(model.parameters(),lr=plan['training']['learning_rate'],weight_decay=plan['training']['weight_decay'])
        x=dict(image=torch.rand(b,3,128,128,device='cuda'),wrist_image=torch.rand(b,3,128,128,device='cuda'),state=torch.rand(b,8,device='cuda'),
               token_ids=torch.ones(b,32,dtype=torch.long,device='cuda'),language_mask=torch.ones(b,32,dtype=torch.bool,device='cuda'))
        target=torch.zeros(b,10,7,device='cuda');mask=torch.ones(b,10,dtype=torch.bool,device='cuda');times=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(25):
            torch.cuda.synchronize();tick=time.monotonic();opt.zero_grad(set_to_none=True)
            loss=masked_mse(model(**x),target,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            torch.cuda.synchronize()
            if i>=5:times.append(time.monotonic()-tick)
        model.eval();torch.cuda.synchronize();tick=time.monotonic()
        with torch.no_grad():
            for _ in range(10):model(**x)
        torch.cuda.synchronize();eval_step=(time.monotonic()-tick)/10
        records[variant]={'train_step_seconds':float(np.mean(times)),'eval_step_seconds':eval_step,'parameters':sum(p.numel() for p in model.parameters()),
                          'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                          'estimated_50epoch_seconds':18400*np.mean(times)+2350*eval_step+240}
        del model,opt,x,target,mask,loss;torch.cuda.empty_cache()
    estimate=records['tiny_norm']['estimated_50epoch_seconds']+2*records['resnet_random']['estimated_50epoch_seconds']+1800
    outer=json.loads((run/'status.json').read_text())
    estimate=float(estimate*1.15)
    budget=outer['timeout_seconds']
    elapsed_setup=time.monotonic()-json.loads((run/'pipeline.json').read_text())['pipeline_started_monotonic']
    fits=estimate<=budget-elapsed_setup-120
    report={'status':'PASS' if fits else 'BUDGET_INSUFFICIENT','synthetic_only':True,'weights_discarded':True,'profiles':records,
            'estimated_remaining_pipeline_seconds_with_margin':estimate,'available_stage_seconds':budget-elapsed_setup,'margin':1.15}
    write(run/'throughput_gate.json',report);print(json.dumps(report,indent=2),flush=True)
    if not fits:raise SystemExit(78)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['weights','profile']);p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run)
    if a.mode=='weights':weights(plan,run)
    else:profile(plan,run)
if __name__=='__main__':main()
