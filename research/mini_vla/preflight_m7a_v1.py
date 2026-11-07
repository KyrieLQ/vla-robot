"""Bounded M7a full-pipeline throughput gate; synthetic weights are discarded."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from .spatial_tokens_model_v1 import SpatialTokensMiniVLA as AttentionMiniVLA,SpatialTokensConfig as AttentionConfig
from .model import masked_mse
from .experiment_v1 import ROOT,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run);pre=json.loads((ROOT/plan['preprocessing']).read_text())
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=AttentionMiniVLA(AttentionConfig(len(pre['vocabulary']),horizon=10)).cuda().train();b=plan['training']['batch_size'];opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    x=dict(image=torch.rand(b,3,128,128,device='cuda'),wrist_image=torch.rand(b,3,128,128,device='cuda'),state=torch.rand(b,8,device='cuda'),token_ids=torch.ones(b,32,dtype=torch.long,device='cuda'),language_mask=torch.ones(b,32,dtype=torch.bool,device='cuda'))
    y=torch.zeros(b,10,7,device='cuda');mask=torch.ones(b,10,dtype=torch.bool,device='cuda');times=[]
    for i in range(25):
        torch.cuda.synchronize();s=time.monotonic();opt.zero_grad(set_to_none=True);loss=masked_mse(model(**x),y,mask);assert torch.isfinite(loss);loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(gradient_norm);opt.step();torch.cuda.synchronize()
        if i>=5:times.append(time.monotonic()-s)
    model.eval();s=time.monotonic()
    with torch.no_grad():
        for _ in range(10):model(**x)
    torch.cuda.synchronize();ev=(time.monotonic()-s)/10
    estimate=float((94950*np.mean(times)+math.ceil(30465/b)*52*ev+480+600)*1.15)
    elapsed=time.monotonic()-json.loads((run/'pipeline.json').read_text())['pipeline_started_monotonic'];available=json.loads((run/'status.json').read_text())['timeout_seconds']-elapsed-120
    report=dict(status='PASS' if estimate<=available else 'BUDGET_INSUFFICIENT',train_step_seconds=float(np.mean(times)),validation_batch_seconds=ev,estimated_full_pipeline_seconds=estimate,available_stage_seconds=available,parameters=sum(p.numel() for p in model.parameters()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,synthetic_weights_discarded=True)
    write(run/'throughput_gate.json',report);print(json.dumps(report),flush=True)
    return 0 if estimate<=available else 78

if __name__=='__main__':raise SystemExit(main())
