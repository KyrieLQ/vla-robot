"""Audit completed M3 work before resuming only the missing evaluation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/mini_vla_m3_20261019_run2'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.monotonic();torch.set_num_threads(4)
    plan=ROOT/'configs/mini_vla_m3_20261019_v2.json'; ph=digest(plan)
    for n,h in json.loads((RUN/'source_manifest.json').read_text()).items(): assert digest(ROOT/n)==h,n
    expected={(t,i) for t in range(10) for i in [2,12,22,32,42]}
    baseline=json.loads((ROOT/'results/mini_vla_m2_20261019_run1/h5/result.json').read_text())
    base={x['pair_id']:x for x in baseline['episodes']}
    summary={}; hashes={}; deltas={}
    def bind(p): hashes[str(p.relative_to(ROOT))]=digest(p)
    for variant in ['tiny_norm','resnet_random','resnet_pretrained']:
        d=RUN/variant/'training';report=json.loads((d/'report.json').read_text());history=json.loads((d/'history.json').read_text())
        assert report['epochs']==50 and report['updates']==18400 and len(history)==50
        for i,row in enumerate(history,1):
            assert row['epoch']==i and row['updates']==i*368 and row['train_frames']==46982
            assert row['valid_target_steps']==452540 and row['validation']['frames']==5988
            assert row['validation']['valid_target_steps']==57720
        chosen=min(history,key=lambda row:row['validation']['normalized_mse'])
        assert chosen['epoch']==report['best_epoch']
        assert abs(chosen['validation']['normalized_mse']-report['selected_validation']['normalized_mse'])<1e-6
        bind(d/'report.json');bind(d/'history.json')
        initial=None;best=None
        for name in ['initial','best','latest']:
            p=d/(name+'.pt');bind(p);assert digest(p)==report['checkpoint_sha256'][name]
            ck=torch.load(p,map_location='cpu',weights_only=False)
            assert ck['plan_sha256']==ph and ck['seed']==42
            assert ck['epoch']==(0 if name=='initial' else report['best_epoch'] if name=='best' else 50)
            if name=='initial':initial=ck['state_dict']
            if name=='best':best=ck['state_dict']
            del ck
        visual=[n for n in initial if n.startswith('vision.') and initial[n].is_floating_point() and n.endswith('weight')]
        assert visual and any(not torch.equal(initial[n],best[n]) for n in visual)
        deltas[variant]={'visual_weight_tensors':len(visual),'changed_at_selected_checkpoint':sum(not torch.equal(initial[n],best[n]) for n in visual)}
        del initial,best
        summary[variant]={k:report[k] for k in ['epochs','updates','best_epoch','training_elapsed_seconds']}
        ev=RUN/variant/'h5/result.json'
        if not ev.exists():
            assert variant=='resnet_pretrained' and not list((RUN/variant/'h5').glob('*/episode.json'))
            summary[variant]['evaluation']='NOT_STARTED';continue
        bind(ev);rows=json.loads(ev.read_text())['episodes'];assert len(rows)==50
        assert {(x['task_id'],x['initial_state_index']) for x in rows}==expected
        wins=losses=0;render=[]
        for row in rows:
            ep=ev.parent/row['pair_id'];bind(ep/'episode.json');bind(ep/'trace.npz');bind(ep/'external.mp4')
            assert json.loads((ep/'episode.json').read_text())==row
            assert row['trace_sha256']==digest(ep/'trace.npz') and row['video_sha256']==digest(ep/'external.mp4')
            assert row['plan_sha256']==ph and row['checkpoint_sha256']==report['checkpoint_sha256']['best']
            with np.load(ep/'trace.npz') as z, np.load(ROOT/'results/mini_vla_m2_20261019_run1/h5'/row['pair_id']/'trace.npz') as b:
                assert z['actions'].shape==(row['steps'],7) and np.isfinite(z['actions']).all()
                assert np.max(np.abs(z['actions']))<=1
                assert bool(z['done'].any())==row['task_success'] and not z['done'][:-1].any()
                assert row['task_success'] or row['steps']==220
                np.testing.assert_array_equal(z['call_steps'],np.arange(0,row['steps'],5))
                np.testing.assert_array_equal(z['action_chunks'][:,:5].reshape(-1,7)[:row['steps']],z['actions'])
                for key in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(z[key],b[key],atol=1e-7,rtol=0)
            ref=base[row['pair_id']];delta=int(row['task_success'])-int(ref['task_success']);wins+=delta==1;losses+=delta==-1
            if row['initial_image_sha256']!=ref['initial_image_sha256']:render.append(row['pair_id'])
        summary[variant].update(evaluated=50,successes=sum(x['task_success'] for x in rows),paired_wins_vs_m2_h5=wins,paired_losses_vs_m2_h5=losses,initial_render_mismatch=render)
    for name in ['weights.json','simulation_preflight/result.json','source_manifest.json']:bind(RUN/name)
    out=RUN/'review';out.mkdir(exist_ok=True)
    result=dict(status='PARTIAL_ACCEPTED_FOR_EVAL_ONLY_RESUME',training_complete=3,evaluation_complete=2,new_evaluation_remaining=50,
                summary=summary,visual_parameter_changes=deltas,bound_files=hashes,original_plan_sha256=ph,
                limitations='All histories, checkpoint hashes, selected epochs and 100 action queues verified; full independent validation inference, final third comparison and video interpretation pending.',elapsed_seconds=time.monotonic()-start)
    p=out/'partial_acceptance.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bound_files'},indent=2))

if __name__=='__main__':main()
