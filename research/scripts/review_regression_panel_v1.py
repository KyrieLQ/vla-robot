"""Independent600-case audit and suite-stratified task-cluster paired statistics."""
import argparse,hashlib,json,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from mini_vla.paired_eval_journal_v1 import load as load_journal
from mini_vla.regression_panel_contract_v1 import case_key,validate_metadata
from mini_vla.train_vlm_head_v1 import sha,write
R=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    pp=Path(a.plan).absolute();p=json.loads(pp.read_text());run=Path(a.run).absolute();out=run/'review';out.mkdir(exist_ok=True);began=time.monotonic()
    for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
    panel=json.loads((R/p['panel']).read_text());arms=p['evaluation']['arm_order'];source=json.loads((run/'result.json').read_text())
    assert source['expected']==len(source['episodes'])==600
    assert json.loads((run/'release_evaluation.json').read_text())['scientific_plan_sha256']==sha(pp)
    gates=json.loads((run/'parallel_gates.json').read_text());assert gates['status']=='PASS' and 1<=gates['workers']<=4
    for gate in gates['online_contracts']:
        assert gate['status']=='PASS' and gate['samples_per_arm']==120 and gate['all_features_bitwise_equal'] and gate['maximum_action_error']<=1e-6
        validate_metadata(gate['metadata'],p)
    prep=json.loads((R/p['preprocessing']).read_text());norm=prep['normalization']['actions']
    table={name:[] for name in arms};rows=[];video_rows=[];latencies={name:[] for name in arms};identities={};total_steps=total_calls=0
    source_rows={(row['arm'],row['pair_id']):row for row in source['episodes']};assert len(source_rows)==600
    for case in panel['cases']:
        pair=case_key(case)
        for arm in arms:
            folder=run/arm/pair;assert folder.is_symlink()
            row=json.loads((folder/'episode.json').read_text());assert row==source_rows[(arm,pair)] and row['case']==case
            assert row['arm']==arm and row['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256'] and row['plan_sha256']==sha(pp)
            assert row['action_horizon']==10 and row['execute_steps']==5 and row['max_steps']==panel['max_action_steps'][case['suite']]
            assert sha(folder/'trace.npz')==row['trace_sha256'] and sha(folder/'trajectory.jsonl')==row['journal_sha256']
            trace=np.load(folder/'trace.npz');calls,steps=load_journal(folder/'trajectory.jsonl');n=len(steps)
            assert n==row['steps'] and 0<n<=row['max_steps'] and len(calls)==row['policy_calls']==(n+4)//5
            assert trace['actions'].shape==(n,7) and trace['robot_state'].shape==(n,8)
            assert list(calls)==list(range(0,n,5))==trace['call_steps'].tolist()
            assert trace['done'].dtype==np.bool_ and not trace['done'][:-1].any() and bool(trace['done'][-1])==row['task_success']
            assert row['task_success'] or n==row['max_steps']
            for i,step in enumerate(steps):
                np.testing.assert_array_equal(trace['actions'][i],step['action']);np.testing.assert_array_equal(trace['robot_state'][i],step['robot_state'])
                assert bool(trace['done'][i])==step['done'] and np.isfinite(step['qpos']).all() and np.isfinite(step['qvel']).all()
            for i,step in enumerate(calls):
                call=calls[step];assert call['arm']==arm and call['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256']
                chunk=trace['action_chunks'][i];standard=trace['normalized_action_chunks'][i]
                np.testing.assert_array_equal(chunk,call['actions']);np.testing.assert_array_equal(standard,call['normalized_actions'])
                command=np.clip(standard*np.asarray(norm['scale'],dtype=np.float32)+np.asarray(norm['mean'],dtype=np.float32),-1,1)
                np.testing.assert_allclose(chunk,command,atol=1e-6,rtol=0)
                for key in ['observation_sha256','vlm_input_sha256','hidden_sha256']:
                    assert trace[key][i]==call[key] and len(call[key])==64
                assert np.isfinite(chunk).all() and np.isfinite(standard).all() and abs(chunk).max()<=1
            identity=row['initial_identity'];assert identity['seed']==case['development_environment_seed'] and identity['history']==case['development_anchor_history']
            for field in ['qpos','qvel']:assert hashlib.sha256(trace['initial_'+field].tobytes()).hexdigest()==identity['initial_'+field+'_sha256']
            if pair in identities:assert identity==identities[pair]
            else:identities[pair]=identity
            selected=case['initial_state_index']==panel['initial_state_indices_per_task'][case['suite']+':'+str(case['task_id'])][0]
            assert selected==row['video_selected']
            if selected:
                video=folder/'external.mp4';assert sha(video)==row['video_sha256'];decoded=0
                reader=imageio.get_reader(video)
                try:
                    for frame in reader:
                        assert frame.shape==(128,128,3);decoded+=1
                finally:reader.close()
                assert decoded==n+1
                video_rows.append(dict(arm=arm,pair_id=pair,path=str(video),frames=decoded,sha256=row['video_sha256']))
            else:assert not (folder/'external.mp4').exists() and row['video_sha256'] is None
            table[arm].append(bool(row['task_success']));latencies[arm].extend(trace['latency'].tolist());total_steps+=n;total_calls+=len(calls)
            rows.append(dict(arm=arm,pair_id=pair,suite=case['suite'],task=case['task_id'],success=row['task_success'],steps=n,policy_calls=len(calls)))
    assert len(rows)==600 and len(identities)==200 and len(video_rows)==120
    rng=np.random.default_rng(panel['statistics']['seed'])
    # Each suite contributes10 resampled tasks; all5 initial states remain a cluster.
    draws=np.stack([rng.integers(0,10,(10000,10))+10*s for s in range(4)],axis=1).reshape(10000,40)
    successes={};per_task={}
    for arm in arms:
        values=np.asarray(table[arm],dtype=float).reshape(40,5);per_task[arm]=values.mean(1)
        successes[arm]=dict(successes=int(values.sum()),episodes=200,rate=float(values.mean()),
          by_suite={suite:dict(successes=int(values[10*i:10*i+10].sum()),episodes=50) for i,suite in enumerate(panel['suites'])},
          task_cluster_descriptive_95_interval=np.quantile(per_task[arm][draws].mean(1),[.025,.975]).tolist(),
          policy_latency_median_seconds=float(np.median(latencies[arm])),policy_latency_p95_seconds=float(np.quantile(latencies[arm],.95)))
    paired={}
    for candidate,control in [('depth2','depth1'),('depth1','existing'),('depth2','existing')]:
        c=np.asarray(table[candidate]);b=np.asarray(table[control]);wins=np.flatnonzero(c&~b);losses=np.flatnonzero(b&~c)
        delta=per_task[candidate]-per_task[control]
        paired[candidate+'_vs_'+control]=dict(rescued=len(wins),new_failures=len(losses),difference=float(c.mean()-b.mean()),
          task_cluster_descriptive_95_interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
          rescued_cases=[case_key(panel['cases'][i]) for i in wins],new_failure_cases=[case_key(panel['cases'][i]) for i in losses],
          by_suite={suite:dict(rescued=int(np.sum((c&~b)[50*i:50*i+50])),new_failures=int(np.sum((b&~c)[50*i:50*i+50]))) for i,suite in enumerate(panel['suites'])})
    selected=sorted(arms,key=lambda arm:(-successes[arm]['successes'],successes[arm]['policy_latency_median_seconds'],p['arms'][arm]['parameters']))[0]
    result=dict(status='PASS',episodes_used=600,expected_episodes=600,distinct_initial_cases=200,tasks=40,suites=4,
      results=successes,paired=paired,selected_system_by_frozen_dev_rule=selected,full_decoded_videos=120,raw_trajectories_verified=600,
      total_physical_steps=total_steps,total_policy_calls=total_calls,all_initial_identities_match=True,worker_gates_verified=gates['workers'],
      new_training_updates=0,final2000_access=False,elapsed_seconds=time.monotonic()-began,
      limitations=['Adaptive development panel,one training seed;intervals descriptive,not unseen-task generalization.',
                  'Newdepth1vsdepth2 is the matched training contrast;existingendpoint has a different training history.',
                  'All journals and traces reviewed;physical states rely on executed guards and recorded evidence,not a second600-episode rerun.',
                  'All120 preselected videos decoded;representative visual inspection still required in followup.'])
    write(out/'independent_acceptance.json',result);write(out/'episodes.json',rows);write(out/'videos.json',video_rows)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
