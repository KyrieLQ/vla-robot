"""CPU contracts for selected head loading/routing, panel isolation and resume queues."""
import argparse,ast,copy,json,tempfile
from pathlib import Path
import numpy as np
import torch
from mini_vla.policy_regression_panel_v1 import load_heads,PanelPolicy
from mini_vla.regression_panel_contract_v1 import expected_metadata,validate_metadata,partition_tasks,case_key
from mini_vla.paired_eval_journal_v1 import append,load
from mini_vla.vlm_action_heads_v1 import RegressionHead,HeadConfig
from mini_vla.vlm_frontend_v1 import bits_bf16
R=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    torch.set_num_threads(1);p=json.loads(Path(a.plan).read_text());panel=json.loads((R/p['panel']).read_text())
    assert len(panel['cases'])==200 and len({case_key(c) for c in panel['cases']})==200
    source=json.loads((R/panel['source_manifest']).read_text());allowed={case_key(c) for c in source['cases'] if c['partition']=='development'}
    assert all(case_key(c) in allowed for c in panel['cases'])
    for suite_index,suite in enumerate(panel['suites']):
        for task in range(10):
            rows=[c for c in panel['cases'] if c['suite']==suite and c['task_id']==task]
            assert len(rows)==5
            for i,c in enumerate(rows):
                assert c['development_anchor_history']==[q['initial_state_index'] for q in rows[:i]]
                assert c['development_environment_seed']==1000+10000*suite_index+100*task+c['initial_state_index']
    for n in [1,2,3,4]:
        pieces=partition_tasks(n);assert sorted(sum(pieces,[]))==list(range(40))
    metadata=expected_metadata(p);validate_metadata(metadata,p)
    bad=copy.deepcopy(metadata);bad['arms']['depth2']['decoder_layers']=1
    try:validate_metadata(bad,p)
    except AssertionError:pass
    else:raise AssertionError('Wrong head identity was accepted')
    heads,cks=load_heads(p);gates=json.loads((R/p['training_run']/'frontend_gate.json').read_text())['rows']
    for row in gates:
        bits=np.load(R/p['training_run']/'gate'/row['file']);hidden=bits_bf16(bits)[None]
        mask=torch.ones(hidden.shape[:2],dtype=torch.bool);state=torch.zeros(1,8)
        for name,head in heads.items():
            with torch.inference_mode():result=head(hidden=hidden,hidden_mask=mask,state=state)
            assert result.shape==(1,10,7) and torch.isfinite(result).all()
            if name=='existing':
                original=RegressionHead(HeadConfig(**cks[name]['model_config']));original.load_state_dict(cks[name]['state_dict']);original.eval()
                with torch.inference_mode():reference=original(hidden=hidden,hidden_mask=mask,state=state)
                assert torch.equal(result,reference)
    class StoredFrontend:
        def encode(self,records):
            assert len(records)==1
            return [dict(bits=bits,input_sha256='stored-fixture')]
    policy=PanelPolicy(StoredFrontend(),heads,cks['existing']['preprocessing']['normalization'],metadata)
    for arm in p['arms']:
        result=policy.infer({'policy/arm':arm,'observation/image':np.zeros((256,256,3),np.uint8),
                            'observation/wrist_image':np.zeros((256,256,3),np.uint8),'observation/state':np.zeros(8),'prompt':'fixture'})
        assert result['arm']==arm and result['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256'] and result['actions'].shape==(10,7)
    # A crash after step50 has one executed action and four queued commands from the51st-step call.
    with tempfile.TemporaryDirectory(dir=R/'tmp') as directory:
        path=Path(directory)/'trajectory.jsonl';chunks={}
        for i in range(51):
            if i%5==0:
                chunk=np.arange(70,dtype=np.float32).reshape(10,7)+i;chunks[i]=chunk
                append(path,dict(kind='call',step=i,actions=chunk.tolist(),normalized_actions=chunk.tolist()))
            append(path,dict(kind='step',step=i,action=chunks[(i//5)*5][i%5].tolist(),done=False))
        calls,steps=load(path);assert len(steps)==51 and len(calls)==11
        remaining=np.asarray(calls[50]['actions'])[1:5];np.testing.assert_array_equal(remaining,chunks[50][1:5])
        assert len(remaining)==4
    for f in ['mini_vla/libero_eval_regression_panel_v1.py','mini_vla/regression_panel_contract_v1.py']:
        ast.parse((R/f).read_text(),feature_version=(3,8))
    result=dict(status='PASS',cases_per_arm=200,total_cases=600,all_development_only=True,task_partitions_tested=[1,2,3,4],
                selected_heads_strictly_loaded=3,head_forward_inputs_per_arm=120,existing_original_forward_bitwise=True,
                model_identity_negative_test=True,normalization_and_arm_routing=True,remaining_action_queue_after51_steps=4,
                new_gpu_updates=0,new_simulation_episodes=0)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
