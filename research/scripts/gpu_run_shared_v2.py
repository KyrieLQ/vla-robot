#!/usr/bin/env python3
"""User-authorized shared GPU admission based on free memory, no process eviction."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
from gpu_run import sample
R=Path(__file__).resolve().parents[1]
def eligible(g,minimum):return g['free_mib']>=minimum

def requirements(plan):
 res=plan['resources'];path=R/res['gpu_sharing_approval'];a=json.loads(path.read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==res['gpu_sharing_approval_sha256']
 assert a['status']=='APPROVED' and a['thread']==plan['thread'] and a['allow_existing_compute_processes'] and a['utilization_is_not_a_veto']
 allowed=set(res['allowed_physical_gpus']);assert allowed.issubset(set(a['allowed_physical_gpus']))
 minimum=int(res['minimum_free_gpu_mib']);count=int(res.get('simultaneous_gpus',4));assert minimum>=a['current_eval_min_free_mib'] and 1<=count<=a['max_simultaneous_gpus']==4
 assert a['prefer_multi_gpu_at_every_launch'] and res.get('multi_gpu_assessment')
 if count==1:assert res['multi_gpu_assessment'].get('single_gpu_reason')
 assert 1<=int(res.get('minimum_gpu_count',count))<=count
 return allowed,minimum,count

def candidates(first,second,allowed,minimum):
 prior={g['uuid'] for g in first if g['index'] in allowed and eligible(g,minimum)}
 return sorted([g for g in second if g['uuid'] in prior and g['index'] in allowed and eligible(g,minimum)],key=lambda g:(-g['free_mib'],g['utilization'],g['index']))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--check-only',action='store_true');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());allowed,minimum,count=requirements(plan)
 path=R/'reports'/('gpu_shared4_selection_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
 record=dict(time=datetime.datetime.now().astimezone().isoformat(),mode='shared_free_memory',minimum_free_mib=minimum,count=count,approval=plan['resources']['gpu_sharing_approval'],existing_compute_processes_allowed=True,exclusive_reservation=False)
 def save():path.write_text(json.dumps(record,indent=2)+'\n')
 try:first=sample();time.sleep(3);second=sample();good=candidates(first,second,allowed,minimum)
 except Exception as e:record.update(status='GPU_QUERY_UNAVAILABLE',error=str(e));save();return 76
 record['samples']=[first,second]
 if len(good)<int(plan['resources'].get('minimum_gpu_count',count)):record['status']='WAITING_FOR_GPU_MEMORY';save();print('WAITING_FOR_GPU_MEMORY',flush=True);return 75
 latest={g['uuid']:g for g in sample()};good=[latest[g['uuid']] for g in good if g['uuid'] in latest and eligible(latest[g['uuid']],minimum)]
 good.sort(key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
 if len(good)<int(plan['resources'].get('minimum_gpu_count',count)):record['status']='WAITING_FOR_GPU_MEMORY';save();return 75
 selected=good[:count]
 record.update(preferred_count=count,actual_count=len(selected),fallback_reason=None if len(selected)==count else 'Fewer qualifying devices; maximum available multiGPU set within preapproved minimum')
 for i,g in enumerate(selected):g['logical_index']=i
 record.update(status='SHARED_MEMORY_ADMITTED',selected=selected);save();print('Selected shared physical GPU(s): '+','.join(str(g['index']) for g in selected),flush=True)
 if a.check_only:return 0
 cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
 if not cmd:ap.error('Command required')
 env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in selected),VLA_GPU_SELECTION=str(path),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
 env.pop('XLA_PYTHON_CLIENT_MEM_FRACTION',None);env.pop('XLA_PYTHON_CLIENT_ALLOCATOR',None)
 code=subprocess.run(cmd,env=env).returncode;record.update(status='COMMAND_FINISHED',returncode=code);save();return code
if __name__=='__main__':raise SystemExit(main())
