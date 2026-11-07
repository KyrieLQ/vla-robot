#!/usr/bin/env python3
"""Two-sample idle GPU selection; no reservation claim and no persistent polling."""
import argparse, csv, datetime, json, os, pathlib, subprocess, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[1]
def sample():
    args=['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    rows=[]
    for r in csv.reader(subprocess.check_output(args,text=True,stderr=subprocess.PIPE,timeout=10).splitlines()):
        r=[v.strip() for v in r]
        rows.append(dict(index=int(r[0]),uuid=r[1],pci_bus_id=r[2],name=r[3],total_mib=int(r[4]),used_mib=int(r[5]),free_mib=int(r[6]),utilization=int(r[7]),compute_pids=[]))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,stderr=subprocess.PIPE,timeout=10)
    for r in csv.reader(apps.splitlines()):
        if len(r)!=2:continue
        for gpu in rows:
            if gpu['uuid']==r[0].strip():gpu['compute_pids'].append(int(r[1].strip()))
    for key in ['CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']:
        restriction=os.environ.get(key)
        if restriction is not None and restriction not in ['all']:
            allowed=[x.strip() for x in restriction.split(',') if x.strip()]
            rows=[g for g in rows if str(g['index']) in allowed or any(g['uuid'].startswith(x) for x in allowed if x.startswith('GPU-'))]
    return rows

def idle(g):
    return not g['compute_pids'] and g['used_mib']<=512 and g['free_mib']>=g['total_mib']*.95 and g['utilization']<=5

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='auto');p.add_argument('--gpus',type=int,default=1)
    p.add_argument('--multi-card-reason');p.add_argument('--check-only',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not 1<=a.gpus<=4: p.error('GPU count must be between 1 and 4')
    if a.gpus>1 and not a.multi_card_reason:p.error('Multi-card mode requires a recorded user authorization or single-card failure reason')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output=ROOT/'reports'/('gpu_selection_'+stamp+'.json')
    record={'time':datetime.datetime.now().astimezone().isoformat(),'request':a.gpu,'count':a.gpus,
            'criterion':{'max_used_mib':512,'min_free_fraction':.95,'max_utilization':5,'compute_pids':'none'},
            'exclusive_reservation':False}
    try:
        first=sample();time.sleep(3);second=sample()
    except Exception as e:
        record.update(status='GPU_QUERY_UNAVAILABLE',error=type(e).__name__)
        output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));return 76
    record['samples']=[first,second]
    good={g['uuid'] for g in first if idle(g)}
    candidates=[g for g in second if g['uuid'] in good and idle(g)]
    if a.gpu!='auto':
        wanted=a.gpu.split(',');candidates=[g for g in candidates if str(g['index']) in wanted or g['uuid'] in wanted]
    candidates.sort(key=lambda g:(-g['free_mib'],g['index']))
    if len(candidates)<a.gpus:
        record['status']='WAITING_FOR_IDLE_GPU';output.write_text(json.dumps(record,indent=2)+'\n')
        print('WAITING_FOR_IDLE_GPU: no suitable device in two samples; no job launched.');return 75
    # Recheck immediately before launching; no default-to-zero fallback.
    latest={g['uuid']:g for g in sample()}
    # If the preferred device became busy, automatically choose another device
    # that passed both earlier samples and is still idle now.
    candidates=[latest[g['uuid']] for g in candidates if g['uuid'] in latest and idle(latest[g['uuid']])]
    if len(candidates)<a.gpus:
        record['status']='WAITING_FOR_IDLE_GPU';output.write_text(json.dumps(record,indent=2)+'\n');return 75
    chosen=candidates[:a.gpus]
    for logical,g in enumerate(chosen):g['logical_index']=logical
    record.update(status='IDLE_OBSERVED',selected=chosen,multi_card_reason=a.multi_card_reason)
    output.write_text(json.dumps(record,indent=2)+'\n')
    print('Selected physical GPU(s):',','.join(str(g['index']) for g in chosen),flush=True)
    if a.check_only:return 0
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:p.error('A command is required unless --check-only')
    env=os.environ.copy();env['CUDA_DEVICE_ORDER']='PCI_BUS_ID';env['CUDA_VISIBLE_DEVICES']=','.join(g['uuid'] for g in chosen)
    env['VLA_GPU_SELECTION']=str(output);env['JAX_PLATFORMS']='cuda'
    env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    env.pop('XLA_PYTHON_CLIENT_MEM_FRACTION',None);env.pop('XLA_PYTHON_CLIENT_ALLOCATOR',None)
    result=subprocess.run(command,env=env)
    record.update(status='COMMAND_FINISHED',returncode=result.returncode)
    output.write_text(json.dumps(record,indent=2)+'\n')
    return result.returncode
if __name__=='__main__':raise SystemExit(main())
