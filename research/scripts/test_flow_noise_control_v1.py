"""Real selected CPU head, explicit old-policy identity, deterministic noise and durable journal."""
import ast, hashlib, json, tempfile, time
from pathlib import Path
import numpy as np
import torch
from mini_vla.policy_m9_v1 import M9Policy
from mini_vla.policy_flow_noise_control_v1 import NoiseControlPolicy, noise_identity
from mini_vla.vlm_action_heads_v1 import FlowHead, HeadConfig
from mini_vla.paired_eval_journal_v1 import append,load
R=Path(__file__).resolve().parents[1]
def main():
    started=time.monotonic();torch.set_num_threads(2)
    run=R/'results/mini_vla_m7_fixed_lora_flow_20261029_run1'
    c=torch.load(run/'training/best.pt',map_location='cpu',weights_only=False)
    head=FlowHead(HeadConfig(**c['model_config']));head.load_state_dict(c['state_dict'])
    bits=np.load(run/'gate/input_000.npy')
    class Front:
        def encode(self,records):return [dict(bits=bits,input_sha256='cpu_test_only')]
    original=M9Policy(Front(),head,c['preprocessing']['normalization'])
    fresh=NoiseControlPolicy(original,'fresh');fixed=NoiseControlPolicy(original,'episode_fixed')
    obs={'observation/image':np.zeros((256,256,3),np.uint8),'observation/wrist_image':np.zeros((256,256,3),np.uint8),'observation/state':np.array([.4,.1,.9,0,0,1.57,.02,-.02],np.float32),'prompt':'fixed CPU test','sampling/task_id':2,'sampling/initial_state_index':42,'sampling/call_index':0}
    state_before={k:v.clone() for k,v in head.state_dict().items()}
    first=fixed.infer(obs);old0=original.infer(obs)
    np.testing.assert_array_equal(first['normalized_actions'],old0['normalized_actions'])
    for call in [0,1,3,17]:
        x={**obs,'sampling/call_index':call};a=fixed.infer(x);b=fresh.infer(x);old=original.infer(x)
        np.testing.assert_array_equal(b['normalized_actions'],old['normalized_actions'])
        np.testing.assert_array_equal(a['normalized_actions'],first['normalized_actions'])
        np.testing.assert_array_equal(a['shadow_fresh_normalized_actions'],old['normalized_actions'])
        assert a['noise_seed']==first['noise_seed'] and a['source_call_index']==0
        seed=int.from_bytes(hashlib.sha256(f'20261020:2:42:{call}'.encode()).digest()[:8],'little')%(2**63-1)
        assert b['noise_seed']==seed
    torch.randn(317);np.testing.assert_array_equal(fixed.infer(obs)['actions'],first['actions'])
    assert noise_identity(2,42,7,'fresh')[1]!=noise_identity(2,42,7,'episode_fixed')[1]
    for ids in [(True,42,0),(2,-1,0),(2,42,1.5)]:
        try:noise_identity(*ids,'episode_fixed')
        except ValueError:pass
        else:raise AssertionError(ids)
    for k,v in head.state_dict().items():assert torch.equal(v,state_before[k])
    with tempfile.TemporaryDirectory(dir=R/'reports') as directory:
        path=Path(directory)/'journal.jsonl';row=dict(kind='call',step=0,actions=first['actions'].tolist(),normalized_actions=first['normalized_actions'].tolist(),shadow_fresh_actions=first['shadow_fresh_actions'].tolist(),noise_seed=first['noise_seed'])
        append(path,row);calls,steps=load(path);assert not steps and calls[0]==row
    old=(R/'mini_vla/libero_eval_fixed_lora_flow_v1.py').read_text();new=(R/'mini_vla/libero_eval_flow_noise_control_v1.py').read_text()
    assert old[old.index('    def reset('):old.index("    if a.variant=='preflight':")]==new[new.index('    def reset('):new.index("    if a.variant=='preflight':")]
    sources=list((R/'mini_vla').glob('*flow_noise*.py'))+list((R/'scripts').glob('*flow_noise*.py'))
    for f in sources:ast.parse(f.read_text())
    result=dict(status='PASS',checks=['call0_exact_original_policy_identity','fresh_mode_exact_original_policy_identity','episode_fixed_independent_of_call_index_and_global_rng','shadow_matches_original_fresh_on_same_observation','private_cpu_noise_hash_identity','invalid_identity_rejected','weights_unchanged','pending_call_journal_roundtrip','canonical_reset_source_unchanged','sources_parse'],elapsed_seconds=time.monotonic()-started,new_optimizer_updates=0,new_simulation_episodes=0,device='cpu')
    (R/'reports/flow_noise_control_cpu_contract_20261103.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
