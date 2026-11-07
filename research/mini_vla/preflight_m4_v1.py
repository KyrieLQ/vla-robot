"""Discarded synthetic throughput estimate; no demonstration training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .attention_model_v1 import AttentionMiniVLA,AttentionConfig
from .model import masked_mse
from .experiment_v1 import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=AttentionMiniVLA(AttentionConfig(28,horizon=10)).cuda().train();b=128
    x=dict(image=torch.rand(b,3,128,128,device='cuda'),wrist_image=torch.rand(b,3,128,128,device='cuda'),state=torch.rand(b,8,device='cuda'),token_ids=torch.ones(b,32,dtype=torch.long,device='cuda'),language_mask=torch.ones(b,32,dtype=torch.bool,device='cuda'))
    y=torch.zeros(b,10,7,device='cuda');mask=torch.ones(b,10,dtype=torch.bool,device='cuda');opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);times=[]
    for i in range(25):
        torch.cuda.synchronize();s=time.monotonic();opt.zero_grad(set_to_none=True);loss=masked_mse(model(**x),y,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();torch.cuda.synchronize()
        if i>=5:times.append(time.monotonic()-s)
    model.eval();tick=time.monotonic()
    with torch.no_grad():
        for _ in range(10):model(**x)
    torch.cuda.synchronize();eval_step=(time.monotonic()-tick)/10
    estimate=float((18400*np.mean(times)+2350*eval_step+240+600)*1.15)
    available=json.loads((run/'status.json').read_text())['timeout_seconds']-(time.monotonic()-json.loads((run/'pipeline.json').read_text())['pipeline_started_monotonic'])-120
    report=dict(status='PASS' if estimate<=available else 'BUDGET_INSUFFICIENT',synthetic_only=True,weights_discarded=True,train_step_seconds=float(np.mean(times)),eval_step_seconds=eval_step,estimated_remaining_pipeline_seconds=estimate,available_seconds=available,parameters=sum(p.numel() for p in model.parameters()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write(run/'throughput_gate.json',report);print(json.dumps(report),flush=True)
    return 0 if estimate<=available else 78

if __name__=='__main__':raise SystemExit(main())
