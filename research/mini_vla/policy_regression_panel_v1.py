"""Three immutable regression heads, sharing one immutable adapted VLM per GPU."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from .policy_m8_v1 import M8Policy, R, sha
from .policy_paired_v1 import AdaptedFrontend
from .regression_depth_pair_v1 import RegressionDepthHead
from .regression_panel_contract_v1 import expected_metadata, validate_metadata
from .vlm_action_heads_v1 import HeadConfig
from .vlm_frontend_v1 import decode_record, bits_bf16
from .data import normalize

def load_heads(plan, device='cpu'):
    heads, checkpoints = {}, {}
    for name, selected in plan['arms'].items():
        path = R/selected['checkpoint']
        assert sha(path) == selected['checkpoint_sha256']
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck.get('epoch') == selected['epoch'] and ck['updates'] == selected['updates']
        if name == 'existing':
            assert ck['architecture'] == 'smolvlm_paired_regression' and ck['arm'] == 'lora'
            assert selected['checkpoint_sha256'] == plan['backbone_checkpoint_sha256']
        else:
            assert ck['architecture'] == 'fixed_adapted_vlm_regression_depth'
            assert ck['decoder_layers'] == selected['decoder_layers']
            assert ck['frontend_checkpoint_sha256'] == plan['backbone_checkpoint_sha256']
            assert ck['vlm_and_adapters_frozen']
        head = RegressionDepthHead(HeadConfig(**ck['model_config']), selected['decoder_layers'])
        head.load_state_dict(ck['state_dict'], strict=True)
        assert sum(x.numel() for x in head.parameters()) == selected['parameters']
        heads[name] = head.to(device).eval().requires_grad_(False)
        checkpoints[name] = ck
    assert all(ck['preprocessing']['normalization'] == checkpoints['existing']['preprocessing']['normalization']
               for ck in checkpoints.values())
    return heads, checkpoints

class PanelPolicy:
    def __init__(self, frontend, heads, normalization, metadata):
        self.frontend = frontend
        self.policies = {name: M8Policy(frontend, head, normalization) for name, head in heads.items()}
        self.metadata = metadata

    def infer(self, observation):
        arm = observation['policy/arm']
        assert arm in self.policies
        result = self.policies[arm].infer(observation)
        result.update(arm=arm, checkpoint_sha256=self.metadata['arms'][arm]['checkpoint_sha256'])
        return result

@torch.inference_mode()
def check(plan, policy, output):
    started = time.monotonic()
    samples = json.loads((R/plan['sample_manifest']).read_text())['samples']
    split = json.loads((R/plan['split']).read_text())
    allowed = {e['episode_id'] for e in split['train']} - {e['episode_id'] for e in split['validation']}
    gate = json.loads((R/plan['training_run']/'frontend_gate.json').read_text())
    index = {e['episode_id']: e for e in json.loads((R/plan['cache']/'train/index.json').read_text())}
    states = np.load(R/plan['numeric_cache']/'train/state.npy', mmap_mode='r')
    assert len(samples) == len(gate['rows']) == 120
    reports, latency = [], {name: [] for name in policy.policies}
    for i, sample in enumerate(samples):
        eid, frame = sample['episode_id'], sample['frame_index']
        assert eid in allowed
        source = R/plan['dataset_root']/('data/chunk-%03d/episode_%06d.parquet' % (eid//1000, eid))
        row = pq.read_table(source, columns=['image','wrist_image','state','episode_index','frame_index'], use_threads=False).slice(frame, 1).to_pylist()[0]
        assert (row['episode_index'], row['frame_index']) == (eid, frame)
        record = decode_record(row, sample['instruction'])
        saved = gate['rows'][i]
        assert (saved['episode_id'], saved['frame_index']) == (eid, frame)
        file = R/plan['training_run']/'gate'/saved['file']
        assert sha(file) == saved['file_sha256']
        stored = np.load(file)
        assert stored.dtype == np.uint16
        feature_hash = hashlib.sha256(stored.tobytes()).hexdigest()
        hashes = json.loads((R/plan['cache']/index[eid]['directory']/'input_hashes.json').read_text())
        input_hash = hashes[frame]
        if isinstance(input_hash, dict):
            input_hash = input_hash['input_sha256']
        hidden = bits_bf16(stored)[None].cuda()
        state = torch.from_numpy(np.array(states[index[eid]['start']+frame]))[None].cuda()
        for name, arm_policy in policy.policies.items():
            observation = {'policy/arm': name, 'observation/image': np.asarray(record['images'][0]),
                           'observation/wrist_image': np.asarray(record['images'][1]),
                           'observation/state': np.asarray(row['state']), 'prompt': sample['instruction']}
            tick = time.monotonic()
            result = policy.infer(observation)
            latency[name].append(time.monotonic()-tick)
            assert result['vlm_input_sha256'] == input_hash and result['hidden_sha256'] == feature_hash
            np.testing.assert_allclose(normalize(np.asarray(row['state']), arm_policy.normalization['state']).numpy(), state[0].cpu().numpy(), atol=1e-6, rtol=0)
            expected = arm_policy.head(hidden=hidden, hidden_mask=torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device), state=state)[0].cpu().numpy()
            stats = arm_policy.normalization['actions']
            command = np.clip(expected*np.asarray(stats['scale'], dtype=np.float32)+np.asarray(stats['mean'], dtype=np.float32), -1, 1)
            np.testing.assert_allclose(result['normalized_actions'], expected, atol=1e-6, rtol=0)
            np.testing.assert_allclose(result['actions'], command, atol=1e-6, rtol=0)
            reports.append(dict(arm=name, episode_id=eid, frame_index=frame, input_sha256=input_hash,
                                feature_sha256=feature_hash, action_max_error=float(np.max(np.abs(expected-result['normalized_actions'])))))
    result = dict(status='PASS', samples_per_arm=120, arms=3, comparisons=len(reports),
                  all_features_bitwise_equal=True, maximum_action_error=max(x['action_max_error'] for x in reports),
                  frontend_trainable_parameters=sum(p.numel() for p in policy.frontend.model.parameters() if p.requires_grad),
                  inference_seconds_p95={k:float(np.quantile(v,.95)) for k,v in latency.items()},
                  metadata=policy.metadata, elapsed_seconds=time.monotonic()-started, rows=reports,
                  new_simulation_episodes=0, new_optimizer_updates=0)
    assert result['frontend_trainable_parameters'] == 0
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--contract-output');parser.add_argument('--metadata-output')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text())
    if args.metadata_output:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        metadata=WebsocketClientPolicy('127.0.0.1',args.port).get_server_metadata()
        validate_metadata(metadata,plan)
        Path(args.metadata_output).write_text(json.dumps(dict(status='PASS',metadata=metadata,new_policy_calls=0),indent=2)+'\n')
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    heads,checkpoints=load_heads(plan,'cuda')
    front=AdaptedFrontend(R/plan['model_directory'],checkpoints['existing']['adapters'])
    policy=PanelPolicy(front,heads,checkpoints['existing']['preprocessing']['normalization'],expected_metadata(plan))
    check(plan,policy,args.contract_output)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,metadata=policy.metadata).serve_forever()

if __name__=='__main__':main()
