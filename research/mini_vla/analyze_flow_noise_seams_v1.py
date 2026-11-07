"""Per-episode noise counterfactual and overlapping-action diagnostics."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .paired_eval_journal_v1 import load

R=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());run=Path(a.run);rows=[]
    norm=json.loads((R/p['preprocessing']).read_text())['normalization']['actions']
    scale=np.array(norm['scale'],np.float32);mean=np.array(norm['mean'],np.float32)
    for task in range(10):
        for init in [2,12,22,32,42]:
            pair=f'task{task}_init{init}';baseline=R/p['fresh_flow_evaluation']/pair;folder=run/'flow/h5'/pair
            old=np.load(baseline/'trace.npz');new=np.load(folder/'trace.npz');calls,steps=load(folder/'trajectory.jsonl')
            # Same initial observation and same first noise are a positive identity control.
            np.testing.assert_allclose(old['normalized_action_chunks'][0],new['normalized_action_chunks'][0],atol=1e-6,rtol=0)
            sameobs=[];locked_overlap=[];fresh_overlap=[];grip=[];previous=None
            for j,(step,c) in enumerate(sorted(calls.items())):
                assert c['noise_mode']=='episode_fixed' and c['source_call_index']==0
                seed=int.from_bytes(hashlib.sha256(f'20261020:{task}:{init}:{j}'.encode()).digest()[:8],'little')%(2**63-1)
                noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
                assert c['shadow_fresh_noise_seed']==seed and c['shadow_fresh_noise_sha256']==hashlib.sha256(noise.numpy().tobytes()).hexdigest()
                locked=np.array(c['normalized_actions']);fresh=np.array(c['shadow_fresh_normalized_actions'])
                raw=np.array(c['actions']);shadow=np.array(c['shadow_fresh_actions'])
                assert locked.shape==fresh.shape==raw.shape==shadow.shape==(10,7) and np.isfinite(fresh).all()
                np.testing.assert_allclose(shadow,np.clip(fresh*scale+mean,-1,1),atol=1e-6,rtol=0)
                if j==0:np.testing.assert_array_equal(locked,fresh)
                else:
                    sameobs.append(float(np.mean((locked[:5]-fresh[:5])**2)))
                    grip.append(float(np.mean((raw[:5,6]>=0)!=(shadow[:5,6]>=0))))
                    locked_overlap.append(float(np.mean((previous[5:]-locked[:5])**2)))
                    fresh_overlap.append(float(np.mean((previous[5:]-fresh[:5])**2)))
                previous=locked
            def seam(trace):
                chunks=trace['normalized_action_chunks']
                return float(np.mean((chunks[:-1,5:]-chunks[1:,:5])**2)) if len(chunks)>1 else None
            def boundary(trace):
                steps=trace['call_steps'][1:]
                if not len(steps):return dict(normalized_mse=None,raw_per_dim_mse=None,gripper_switch_rate=None)
                delta=trace['actions'][steps]-trace['actions'][steps-1]
                return dict(normalized_mse=float(np.mean((delta/scale)**2)),raw_per_dim_mse=np.mean(delta**2,axis=0).tolist(),
                            gripper_switch_rate=float(np.mean((trace['actions'][steps,6]>=0)!=(trace['actions'][steps-1,6]>=0))))
            rows.append(dict(task_id=task,init=init,policy_calls=len(calls),first_call_matches_baseline=True,
                             same_observation_noise_mse=float(np.mean(sameobs)) if sameobs else None,
                             same_observation_gripper_disagreement=float(np.mean(grip)) if grip else None,
                             fixed_overlap_mse=float(np.mean(locked_overlap)) if locked_overlap else None,
                             shadow_fresh_overlap_mse=float(np.mean(fresh_overlap)) if fresh_overlap else None,
                             baseline_path_overlap_mse=seam(old),fixed_path_overlap_mse=seam(new),
                             baseline_executed_boundary=boundary(old),fixed_executed_boundary=boundary(new)))
    keys=[k for k in rows[0] if k.endswith('mse') or k.endswith('disagreement')]
    result=dict(status='PASS',episodes=50,rows=rows,episode_equal_weight_means={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys},
                limitations='Same-observation shadow actions isolate immediate noise effects, not counterfactual future success. Trajectory overlap statistics also reflect changed observations and path lengths. Reduced action variation alone does not imply improvement.',
                shadow_outputs_executed=False,new_optimizer_updates=0)
    (run/'noise_seam_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
