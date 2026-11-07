"""Positive paths and deliberate corruptions of the recorded-I/O acceptance gates."""
import datetime,json,os,pathlib,tempfile
import numpy as np
from paired_live_audit_v1 import (check_saved_branch,replay_model_inputs,check_physical,image_hash,arrays,METHODS)
from paired_live_analyze_v1 import robust_rescue_ids
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

class ToyPolicy:
    def __init__(self,delta=0.0):self.delta=delta
    def infer(self,request):
        assert set(request)<={'observation/image','observation/wrist_image','observation/state','prompt','_policy_seed','_policy_rng_advance'}
        marker=float(request['observation/image'][0,0,0])
        return {'actions':np.full((10,7),marker/1000+self.delta)}

def fixture(directory,method):
    plan={'max_total_steps':220,'fine_steps':20,'retreat_steps':5,'retreat_action':[0,0,.3,0,0,0,-1]}
    root={'root_id':'synthetic_t60','branch_step':60,'policy_seed':42,'resample_seed':314}
    queue=[];calls=[];chunks=[];actions=[];hashes=[];ext=[];wr=[]
    for i in range(160):
        image=np.full((224,224,3),(i+1)%256,dtype=np.uint8);wrist=np.flip(image,axis=0).copy()
        hashes.append(image_hash(image,wrist))
        if method=='retreat5_fine20' and i<5:action=np.asarray(plan['retreat_action'])
        else:
            if not queue:
                prediction=ToyPolicy().infer({'observation/image':image})['actions'];chunks.append(prediction);calls.append(i+60)
                ext.append(image);wr.append(wrist)
                fine=(method=='fine20' and i<20) or (method=='retreat5_fine20' and i<25)
                queue.extend(prediction[:1 if fine else 5])
            action=queue.pop(0)
        actions.append(action.copy())
    hashes.append(hashes[-1]);count=len(calls)
    row={'root_id':root['root_id'],'method':method,'steps':160,'policy_calls':count,'language':'synthetic diagnostic only',
         'task_success':False,'first_success_step':None,'root_image_sha256':hashes[0]}
    tr={'action':np.array(actions),'action_chunk':np.array(chunks),'robot_state':np.zeros((161,8)),
        'image_sha256':np.array(hashes),'done':np.zeros(160,dtype=bool),'call_step':np.array(calls),'inference_seconds':np.zeros(count)}
    obs={'external':np.array(ext),'wrist':np.array(wr),'robot_state':np.zeros((count,8)),'call_step':np.array(calls)}
    oracle={'goal_predicate':np.zeros((161,1),dtype=bool)}
    oracle.update({k:np.zeros((161,3)) for k in ('qpos','qvel','control','object_position','object_quaternion')})
    meta={'prompt':row['language'],'policy_seed':314 if method=='resample5' else 42,'policy_rng_advance':0 if method=='resample5' else 12}
    directory.mkdir();np.savez_compressed(directory/'policy_trace.npz',**tr);np.savez_compressed(directory/'policy_inputs.npz',**obs)
    np.savez_compressed(directory/'oracle.npz',**oracle);(directory/'policy_inputs.json').write_text(json.dumps(meta))
    return row,root,plan

def main():
    tests=[]
    def reject(name,call):
        try:call()
        except (AssertionError,ValueError):tests.append({'name':name,'status':'PASS_REJECTED_CORRUPTION'});return
        raise AssertionError('Corruption was accepted: '+name)
    with tempfile.TemporaryDirectory(prefix='paired-live-cpu-',dir=ROOT/'tmp') as tmp:
        tmp=pathlib.Path(tmp)
        for method in METHODS:
            d=tmp/method;row,root,plan=fixture(d,method)
            audit=replay_model_inputs(ToyPolicy(),d,row,root,plan)
            assert audit['max_action_chunk_error']==0
            tests.append({'name':'positive_cadence_and_input_replay_'+method,'status':'PASS','calls':audit['calls_checked']})
        d=tmp/'continue_a';row,root,plan=fixture(tmp/'fault_fixture','continue_a');d=tmp/'fault_fixture'
        reject('prediction_difference_1e-4',lambda:replay_model_inputs(ToyPolicy(1e-4),d,row,root,plan))
        obs=arrays(d/'policy_inputs.npz');base={k:v.copy() for k,v in obs.items()}
        obs['external'][0,0,0,1]^=1;np.savez_compressed(d/'policy_inputs.npz',**obs)
        reject('single_input_byte_changed',lambda:check_saved_branch(d,row,root,plan))
        np.savez_compressed(d/'policy_inputs.npz',**base)
        tr=arrays(d/'policy_trace.npz');original={k:v.copy() for k,v in tr.items()}
        tr['action'][0,0]+=1e-4;np.savez_compressed(d/'policy_trace.npz',**tr)
        reject('executed_action_differs_from_chunk',lambda:check_saved_branch(d,row,root,plan))
        tr={k:v.copy() for k,v in original.items()};tr['done'][0]=True;np.savez_compressed(d/'policy_trace.npz',**tr)
        reject('goal_predicate_mismatch',lambda:check_saved_branch(d,row,root,plan))
        tr={k:v.copy() for k,v in original.items()};tr['call_step'][1]+=1;obs={k:v.copy() for k,v in base.items()};obs['call_step'][1]+=1
        np.savez_compressed(d/'policy_trace.npz',**tr);np.savez_compressed(d/'policy_inputs.npz',**obs)
        reject('forged_call_schedule_in_both_files',lambda:check_saved_branch(d,row,root,plan))
        np.savez_compressed(d/'policy_trace.npz',**original);np.savez_compressed(d/'policy_inputs.npz',**base)
        meta=json.loads((d/'policy_inputs.json').read_text());meta['policy_seed']+=1;(d/'policy_inputs.json').write_text(json.dumps(meta))
        reject('wrong_rng_seed',lambda:check_saved_branch(d,row,root,plan))
        physical={k:np.zeros(3) for k in ('qpos','qvel','control','object_position','object_quaternion')}
        altered={k:v.copy() for k,v in physical.items()};altered['qpos'][0]=1e-4
        reject('physical_difference_1e-4',lambda:check_physical(altered,physical))
        roots=[{'root_id':f'r{i}','pair_id':f'case{i//2}'} for i in range(16)];rows={}
        for r in roots:
            for m in METHODS:rows[(r['root_id'],m)]={'baseline_success':False,'task_success':m=='fine20'}
        assert len(robust_rescue_ids(roots,rows))==8
        for r in roots[:2]:rows[(r['root_id'],'continue_b')]['task_success']=True
        assert len(robust_rescue_ids(roots,rows))==7
        for r in roots[2:4]:rows[(r['root_id'],'continue_a')]['baseline_success']=True
        assert len(robust_rescue_ids(roots,rows))==6
        tests.append({'name':'robust_rescue_episode_dedup_and_both_control_failure_gate','status':'PASS'})
    report={'status':'PASS','scope':'CPU-only synthetic fault injection; not GPU inference or simulator validation',
            'created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'tests':tests,'gpu_preflight_required':True}
    (ROOT/'reports/paired_live_cpu_checks_20261014.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
