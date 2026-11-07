"""Independent all100 journal/trace/video audit, no model inference or simulation."""
import hashlib,json,time,math
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_m6_paired_dev50_20261029_run3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True);pp=R/'configs/mini_vla_m6_paired_dev50_20261029_v3.json';plan=json.loads(pp.read_text());recorded=json.loads((RUN/'analysis.json').read_text());ph=sha(pp)
 for f,h in plan['input_sha256'].items():
  assert sha(R/f)==h,f
  frozen=RUN/'frozen_source'/f
  if frozen.exists():assert sha(frozen)==h
 assert json.loads((R/plan['training_acceptance']).read_text())['status']=='PASS'
 samples=json.loads((R/plan['sample_manifest']).read_text())['samples'];cache_gate=json.loads((R/plan['cache']/'gate/complete.json').read_text())['comparisons'];contracts={}
 for arm in ['frozen','lora']:
  gate=json.loads((RUN/arm/'online_contract.json').read_text());assert gate['status']=='PASS' and gate['samples']==120 and gate['maximum_action_error']==0 and gate['checkpoint_sha256']==plan['arms'][arm]['checkpoint_sha256']
  assert all(not v for v in gate['loading'].values()) and gate['all_deployed_parameters_frozen']
  for row,s,old in zip(gate['rows'],samples,cache_gate):assert (row['episode_id'],row['frame_index'])==(s['episode_id'],s['frame_index']) and row['input_sha256']==old['input_sha256'] and row['action_max_error']==0
  contracts[arm]=dict(samples=120,maximum_action_error=0,features_reference=gate['features_bitwise_equal_to'])
 assert contracts['frozen']['features_reference']=='original_frozen_cache' and contracts['lora']['features_reference']=='current_adapted_raw_training_interface'
 positive=json.loads((RUN/'simulation_preflight/result.json').read_text());assert positive['status']=='PASS' and positive['success_verified'] and positive['max_qpos_error']==0
 stats=json.loads((R/plan['preprocessing']).read_text())['normalization']['actions'];scale=np.array(stats['scale'],np.float32);mean=np.array(stats['mean'],np.float32)
 expected={(t,i) for t in range(10) for i in [2,12,22,32,42]};allrows={};traces={};movies=[];metrics={};timelines={};finals={};artifacts={}
 for arm,directory in [('original',R/plan['original_evaluation']),('frozen',RUN/'frozen/h5'),('lora',RUN/'lora/h5')]:
  result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS' and len(result['episodes'])==50;rows={};latencies=[];server=[]
  for task,init in sorted(expected):
   key=(task,init);pair=f'task{task}_init{init}';folder=directory/pair;e=json.loads((folder/'episode.json').read_text());assert e in result['episodes'] and (e['task_id'],e['initial_state_index'])==key
   assert sha(folder/'trace.npz')==e['trace_sha256'] and sha(folder/'external.mp4')==e['video_sha256'];artifacts[str((folder/'episode.json').relative_to(R))]=sha(folder/'episode.json')
   with np.load(folder/'trace.npz') as z:tr={k:z[k] for k in z.files}
   n=e['steps'];assert 1<=n<=220 and len(tr['actions'])==len(tr['done'])==n and e['policy_calls']==(n+4)//5
   assert e['task_success']==bool(tr['done'][-1]) and not tr['done'][:-1].any() and (e['task_success'] or n==220)
   np.testing.assert_array_equal(tr['call_steps'],np.arange(0,n,5))
   for step in range(n):np.testing.assert_array_equal(tr['actions'][step],tr['action_chunks'][step//5,step%5])
   np.testing.assert_allclose(tr['action_chunks'],np.clip(tr['normalized_action_chunks']*scale+mean,-1,1),atol=1e-6,rtol=0)
   for f in ['actions','robot_state','normalized_action_chunks','latency','server_ms']:assert np.isfinite(tr[f]).all()
   for f in ['observation_sha256','vlm_input_sha256']:assert len(tr[f])==e['policy_calls'] and all(len(x)==64 for x in tr[f])
   if arm=='original':assert e['task_success'] in [True,False]
   else:
    assert e['checkpoint_sha256']==plan['arms'][arm]['checkpoint_sha256'] and e['plan_sha256']==ph and e['arm']==arm
    assert sha(folder/'trajectory.jsonl')==e['journal_sha256'];call_rows={};step_rows=[]
    for line in (folder/'trajectory.jsonl').read_text().splitlines():
     row=json.loads(line)
     if row['kind']=='call':assert row['step']==len(step_rows) and row['step']%5==0 and row['step'] not in call_rows;call_rows[row['step']]=row
     else:assert row['kind']=='step' and row['step']==len(step_rows);step_rows.append(row)
    assert len(step_rows)==n and set(call_rows)==set(tr['call_steps'].tolist())
    for j,step in enumerate(step_rows):
     np.testing.assert_array_equal(step['action'],tr['actions'][j]);np.testing.assert_array_equal(step['robot_state'],tr['robot_state'][j]);assert step['done']==bool(tr['done'][j]);assert np.isfinite(step['qpos']).all() and np.isfinite(step['qvel']).all()
    for j,s in enumerate(tr['call_steps']):
     row=call_rows[int(s)];np.testing.assert_array_equal(row['actions'],tr['action_chunks'][j]);np.testing.assert_array_equal(row['normalized_actions'],tr['normalized_action_chunks'][j]);assert row['observation_sha256']==tr['observation_sha256'][j] and row['vlm_input_sha256']==tr['vlm_input_sha256'][j]
    base=allrows['original'][key];assert e['initial_image_sha256']==base['initial_image_sha256'] and e['language']==base['language']
    for f in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(tr[f],traces['original'][key][f],atol=1e-7,rtol=0)
    # Every frame decoded; select fixed temporal quantiles for qualitative review.
    selected={int(round(n*q)) for q in [0,.2,.4,.6,.8,1]};frames={};count=0
    with imageio.get_reader(str(folder/'external.mp4'),ffmpeg_params=['-threads','1']) as reader:
     for j,frame in enumerate(reader):
      assert frame.shape==(256,256,3)
      if j in selected:frames[j]=Image.fromarray(frame)
      count+=1
    assert count==n+1;movies.append(dict(arm=arm,pair=pair,frames=count,sha256=e['video_sha256']));timelines[arm,key]=frames;finals[arm,key]=frames[n]
   rows[key]=e;traces.setdefault(arm,{})[key]=tr;latencies.extend(tr['latency'][1:]);server.extend(tr['server_ms'][1:])
  assert set(rows)==expected;allrows[arm]=rows
  metrics[arm]=dict(successes=sum(x['task_success'] for x in rows.values()),episodes=50,success_rate=sum(x['task_success'] for x in rows.values())/50,steps=sum(x['steps'] for x in rows.values()),policy_calls=sum(x['policy_calls'] for x in rows.values()),rpc_median_ms=float(np.median(latencies)*1000),server_median_ms=float(np.median(server)),evaluation_seconds=result['elapsed_seconds'])
  print('verified',arm,metrics[arm],flush=True)
 def compare(left,right):
  values=np.array([int(allrows[right][k]['task_success'])-int(allrows[left][k]['task_success']) for k in sorted(expected)])
  task=np.array([np.mean([int(allrows[right][k]['task_success'])-int(allrows[left][k]['task_success']) for k in expected if k[0]==t]) for t in range(10)])
  indices=np.random.default_rng(20261019).integers(0,10,size=(10000,10));interval=np.quantile(task[indices].mean(1),[.025,.975]).tolist()
  return dict(wins=int((values==1).sum()),losses=int((values==-1).sum()),difference=float(values.mean()),task_cluster_descriptive_95_interval=interval,improved_pairs=[f'task{k[0]}_init{k[1]}' for k in sorted(expected) if allrows[right][k]['task_success'] and not allrows[left][k]['task_success']],harmed_pairs=[f'task{k[0]}_init{k[1]}' for k in sorted(expected) if allrows[left][k]['task_success'] and not allrows[right][k]['task_success']])
 paired=compare('frozen','lora');versus={a:compare('original',a) for a in ['frozen','lora']}
 for k in ['wins','losses','difference']:assert paired[k]==recorded['paired'][k]
 np.testing.assert_allclose(paired['task_cluster_descriptive_95_interval'],recorded['paired']['task_cluster_descriptive_95_interval'],atol=1e-12,rtol=0)
 for arm in metrics:assert metrics[arm]['successes']==recorded['metrics'][arm]['successes']
 tasks=[dict(task_id=t,**{a:sum(x['task_success'] for k,x in allrows[a].items() if k[0]==t) for a in allrows},episodes_per_arm=5) for t in range(10)]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 selected=[k for k in sorted(expected) if not allrows['frozen'][k]['task_success'] or not allrows['lora'][k]['task_success']]+[(0,2)]
 for key in selected:
  canvas=Image.new('RGB',(1200,450),'white');draw=ImageDraw.Draw(canvas)
  for j,arm in enumerate(['frozen','lora']):
   e=allrows[arm][key];draw.text((6,j*225+4),f'{arm} task{key[0]}_init{key[1]} success={e["task_success"]}, steps={e["steps"]}; frames at 0/20/40/60/80/100%',font=font,fill='black')
   for c,(step,im) in enumerate(sorted(timelines[arm,key].items())):canvas.paste(im.resize((196,196)),(c*200,j*225+26))
  canvas.save(out/f'paired_timeline_task{key[0]}_init{key[1]}.png')
 for arm in ['frozen','lora']:
  canvas=Image.new('RGB',(1600,900),'white');draw=ImageDraw.Draw(canvas)
  for j,key in enumerate(sorted(expected)):
   x=j%10*160;y=j//10*180;e=allrows[arm][key];canvas.paste(finals[arm,key].resize((156,156)),(x,y+24));draw.text((x+2,y+2),f'{key[0]}/{key[1]} '+('OK' if e['task_success'] else 'FAIL'),font=font,fill='green' if e['task_success'] else 'red')
  canvas.save(out/(arm+'_all50_final_frames.png'))
 report=dict(status='PASS',episodes=100,metrics=metrics,paired_lora_vs_frozen=paired,versus_original=versus,per_task=tasks,online_contracts=contracts,all100_videos_fully_decoded=True,decoded_frames=sum(x['frames'] for x in movies),videos=movies,all100_queues_and_journals_verified=True,all_initial_images_and_physical_states_match_original=True,source_bindings_verified=len(plan['input_sha256']),artifacts_sha256=artifacts,supervised_elapsed_seconds=json.loads((RUN/'status.json').read_text())['elapsed_seconds'],elapsed_seconds=time.monotonic()-start,new_training_updates=0,new_rollouts=0,new_gpu_compute=0,claim_boundary='One seed,one additional epoch,adaptive originalSpatial50 development; descriptive cluster interval includes zero,not a final2000/generalization or stable superiority claim',limitations=['Raw wrist/input image bytes are not saved at every call; source runtime input hash assertions and stored call hashes are audited, not reconstructed from lossy video.', 'Physical state journal and original simulator success predicate assertions checked; this audit does not rerun100physics episodes.'])
 (out/'independent_acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['videos','artifacts_sha256','per_task']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
