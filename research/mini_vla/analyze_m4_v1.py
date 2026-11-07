"""Paired M4 versus pretrained M3 on the fixed50 development cases."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import sha256
from .experiment_v1 import ROOT,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run);base=ROOT/plan['baseline_run'];assert sha256(base/'review/independent_acceptance.json')==plan['baseline_acceptance_sha256']
    sub=run/'resnet_pretrained';rep=json.loads((sub/'training/report.json').read_text());assert rep['epochs']==50 and rep['updates']==18400
    assert sha256(sub/'training/best.pt')==rep['checkpoint_sha256']['best']
    dirs={'m3':base/'resnet_pretrained/h5','m4':sub/'h5'};rows={};traces={};metrics={}
    expected={(t,i) for t in range(10) for i in [2,12,22,32,42]}
    for v,d in dirs.items():
        result=json.loads((d/'result.json').read_text());assert result['status']=='PASS' and len(result['episodes'])==50
        rows[v]={};traces[v]={};latency=[];calls=steps=0
        for e in result['episodes']:
            k=(e['task_id'],e['initial_state_index']);assert k in expected and k not in rows[v]
            ep=d/e['pair_id'];assert sha256(ep/'trace.npz')==e['trace_sha256'] and sha256(ep/'external.mp4')==e['video_sha256']
            if v=='m4':assert e['checkpoint_sha256']==rep['checkpoint_sha256']['best'] and e['plan_sha256']==sha256(Path(a.plan))
            with np.load(ep/'trace.npz') as z:trace={k:z[k] for k in z.files}
            assert e['task_success']==bool(trace['done'].any()) and not trace['done'][:-1].any()
            assert e['task_success'] or e['steps']==220
            np.testing.assert_array_equal(trace['call_steps'],np.arange(0,e['steps'],5))
            np.testing.assert_array_equal(trace['action_chunks'][:,:5].reshape(-1,7)[:e['steps']],trace['actions'])
            rows[v][k]=e;traces[v][k]=trace;latency.extend(trace['latency'][1:]);calls+=e['policy_calls'];steps+=e['steps']
        assert set(rows[v])==expected
        metrics[v]=dict(successes=sum(e['task_success'] for e in rows[v].values()),episodes=50,calls=calls,steps=steps,rpc_median_ms=float(np.median(latency)*1000))
    diff=[];bytask=[];render=[]
    for k in sorted(expected):
        for f in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(traces['m3'][k][f],traces['m4'][k][f],atol=1e-7,rtol=0)
        if rows['m3'][k]['initial_image_sha256']!=rows['m4'][k]['initial_image_sha256']:render.append(k)
        diff.append(int(rows['m4'][k]['task_success'])-int(rows['m3'][k]['task_success']))
    for t in range(10):bytask.append(np.mean([int(rows['m4'][k]['task_success'])-int(rows['m3'][k]['task_success']) for k in expected if k[0]==t]))
    boot=np.random.default_rng(20261019).choice(bytask,(10000,10),replace=True).mean(1)
    result=dict(status='COMPLETE_REVIEW_REQUIRED',metrics=metrics,paired=dict(wins=diff.count(1),losses=diff.count(-1),difference=float(np.mean(diff)),task_cluster_descriptive_95_interval=np.quantile(boot,[.025,.975]).tolist()),initial_render_mismatch=render,training=rep,claim_boundary='Combined fusion+decoder architecture change, one seed fixed50 development; not full ACT, not isolated attention-only effect, not full benchmark.')
    write(run/'analysis.json',result);print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
