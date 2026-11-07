"""Independent 50-episode noise/journal/queue/video audit, no policy reruns."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();RUN=Path(a.run)
 started=time.monotonic();torch.set_num_threads(2);out=RUN/'review';out.mkdir(exist_ok=True);pp=Path(a.plan);p=json.loads(pp.read_text());ph=sha(pp);recorded=json.loads((RUN/'analysis.json').read_text())
 for f,h in p['input_sha256'].items():
  assert sha(R/f)==h,f
  frozen=RUN/'frozen_source'/f
  if frozen.exists():assert sha(frozen)==h
 assert json.loads((R/p['training_acceptance']).read_text())['status']=='PASS'
 gate=json.loads((RUN/'flow/online_contract.json').read_text());samples=json.loads((R/p['sample_manifest']).read_text())['samples'];assert gate['status']=='PASS' and gate['samples']==120 and len(gate['rows'])==120 and gate['maximum_action_error']==0 and gate['all_features_bitwise_equal'] and gate['noise_seed_and_ten_step_Euler_verified']
 assert gate['checkpoint_sha256']==p['checkpoint_sha256'] and gate['backbone_checkpoint_sha256']==p['backbone_checkpoint_sha256']
 assert gate['frontend_trainable_parameters']==0 and gate['head_parameters']==697287 and all(not x for x in gate['model_loading'].values())
 for s,x in zip(samples,gate['rows']):assert (s['episode_id'],s['frame_index'])==(x['episode_id'],x['frame_index']) and x['normalized_action_max_error']==x['command_max_error']==0
 positive=json.loads((RUN/'simulation_preflight/result.json').read_text());assert positive['status']=='PASS' and positive['success_verified'] and positive['steps']==73 and positive['max_qpos_error']==0
 stats=json.loads((R/p['preprocessing']).read_text())['normalization']['actions'];scale=np.asarray(stats['scale'],np.float32);mean=np.asarray(stats['mean'],np.float32)
 paths={'regression':R/p['baseline_evaluation'],'control_flow':R/p['control_flow_evaluation'],'flow':RUN/'flow/h5'};expected=[(t,i) for t in range(10) for i in [2,12,22,32,42]];allrows={};alltraces={};metrics={};videos=[];frames_store={};artifacts={};noise_calls=0
 for arm,directory in paths.items():
  result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS' and len(result['episodes'])==50;rows={};traces={};latency=[];server=[]
  if arm=='flow':assert result['metadata']['euler_steps']==10 and result['metadata']['backbone_checkpoint_sha256']==p['backbone_checkpoint_sha256']
  for task,init in expected:
   pair=f'task{task}_init{init}';folder=directory/pair;e=json.loads((folder/'episode.json').read_text());assert e in result['episodes'] and (e['task_id'],e['initial_state_index'])==(task,init)
   assert sha(folder/'trace.npz')==e['trace_sha256'] and sha(folder/'external.mp4')==e['video_sha256'];artifacts[str((folder/'episode.json').relative_to(R))]=sha(folder/'episode.json')
   with np.load(folder/'trace.npz') as z:tr={k:z[k] for k in z.files}
   n=e['steps'];calls=(n+4)//5;assert 1<=n<=220 and len(tr['actions'])==len(tr['done'])==n and e['policy_calls']==calls
   assert e['task_success']==bool(tr['done'][-1]) and not tr['done'][:-1].any() and (e['task_success'] or n==220)
   np.testing.assert_array_equal(tr['call_steps'],np.arange(0,n,5))
   for step in range(n):np.testing.assert_array_equal(tr['actions'][step],tr['action_chunks'][step//5,step%5])
   np.testing.assert_allclose(tr['action_chunks'],np.clip(tr['normalized_action_chunks']*scale+mean,-1,1),atol=1e-6,rtol=0)
   for field in ['actions','robot_state','normalized_action_chunks','latency','server_ms']:assert np.isfinite(tr[field]).all()
   for field in ['observation_sha256','vlm_input_sha256']:assert len(tr[field])==calls and all(len(x)==64 for x in tr[field])
   if arm in ['control_flow','flow']:
    assert len(tr['noise_seed'])==len(tr['noise_sha256'])==calls
    for j,(ns,nh) in enumerate(zip(tr['noise_seed'],tr['noise_sha256'])):
     seed=int.from_bytes(hashlib.sha256(f'20261020:{task}:{init}:{j}'.encode()).digest()[:8],'little')%(2**63-1);assert int(ns)==seed
     noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32);assert nh==hashlib.sha256(noise.numpy().tobytes()).hexdigest()
   if arm!='regression':
    base=allrows['regression'][task,init];assert base['initial_image_sha256']==e['initial_image_sha256'] and base['language']==e['language']
    for field in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(tr[field],alltraces['regression'][task,init][field],atol=1e-7,rtol=0)
   if arm=='flow':
    assert e['checkpoint_sha256']==p['checkpoint_sha256'] and e['plan_sha256']==ph and e['arm']=='flow' and sha(folder/'trajectory.jsonl')==e['journal_sha256']
    callrows={};steps=[]
    for line in (folder/'trajectory.jsonl').read_text().splitlines():
     x=json.loads(line)
     if x['kind']=='call':assert x['step']==len(steps) and x['step']%5==0 and x['step'] not in callrows;callrows[x['step']]=x
     else:assert x['kind']=='step' and x['step']==len(steps);steps.append(x)
    assert len(steps)==n and set(callrows)==set(tr['call_steps'].tolist())
    for j,x in enumerate(steps):
     np.testing.assert_array_equal(x['action'],tr['actions'][j]);np.testing.assert_array_equal(x['robot_state'],tr['robot_state'][j]);assert x['done']==bool(tr['done'][j]) and np.isfinite(x['qpos']).all() and np.isfinite(x['qvel']).all()
    for j,step in enumerate(tr['call_steps']):
     x=callrows[int(step)];np.testing.assert_array_equal(x['actions'],tr['action_chunks'][j]);np.testing.assert_array_equal(x['normalized_actions'],tr['normalized_action_chunks'][j]);assert x['observation_sha256']==tr['observation_sha256'][j] and x['vlm_input_sha256']==tr['vlm_input_sha256'][j] and x['noise_seed']==int(tr['noise_seed'][j]) and x['noise_sha256']==tr['noise_sha256'][j]
    noise_calls+=calls
    chosen={int(round(n*f)) for f in [0,.2,.4,.6,.8,1]};frames={};count=0
    with imageio.get_reader(str(folder/'external.mp4'),ffmpeg_params=['-threads','1']) as reader:
     for j,frame in enumerate(reader):
      assert frame.shape==(256,256,3)
      if j in chosen:frames[j]=Image.fromarray(frame)
      count+=1
    assert count==n+1;frames_store[task,init]=frames;videos.append(dict(pair=pair,frames=count,sha256=e['video_sha256']))
   rows[task,init]=e;traces[task,init]=tr;latency.extend(tr['latency'][1:]);server.extend(tr['server_ms'][1:])
  allrows[arm]=rows;alltraces[arm]=traces;metrics[arm]=dict(successes=sum(x['task_success'] for x in rows.values()),episodes=50,success_rate=sum(x['task_success'] for x in rows.values())/50,steps=sum(x['steps'] for x in rows.values()),calls=sum(x['policy_calls'] for x in rows.values()),rpc_median_ms=float(np.median(latency)*1000),server_median_ms=float(np.median(server)),evaluation_seconds=result['elapsed_seconds']);assert metrics[arm]['successes']==recorded['metrics'][arm]['successes'];print('verified',arm,metrics[arm],flush=True)
 def compare(left):
  differences=np.array([int(allrows['flow'][k]['task_success'])-int(allrows[left][k]['task_success']) for k in expected]);task=differences.reshape(10,5).mean(1);idx=np.random.default_rng(20261019).integers(0,10,(10000,10));ci=np.quantile(task[idx].mean(1),[.025,.975]).tolist()
  return dict(wins=int((differences==1).sum()),losses=int((differences==-1).sum()),difference=float(differences.mean()),task_cluster_descriptive_95_interval=ci,improved_pairs=[f'task{k[0]}_init{k[1]}' for k,d in zip(expected,differences) if d==1],harmed_pairs=[f'task{k[0]}_init{k[1]}' for k,d in zip(expected,differences) if d==-1])
 comparisons={a:compare(a) for a in ['regression','control_flow']}
 for arm,c in comparisons.items():
  recorded_pair=recorded['paired_vs_'+arm]
  for k in ['wins','losses','difference']:assert c[k]==recorded_pair[k]
  np.testing.assert_allclose(c['task_cluster_descriptive_95_interval'],recorded_pair['task_cluster_descriptive_95_interval'],atol=1e-12,rtol=0)
 tasks=[dict(task_id=t,**{a:sum(e['task_success'] for k,e in allrows[a].items() if k[0]==t) for a in paths},episodes_per_model=5) for t in range(10)]
 # Deterministic representative selection across improvements, regressions, shared failure/success.
 groups={'new_success_regression_fail':[k for k in expected if allrows['flow'][k]['task_success'] and not allrows['regression'][k]['task_success']], 'new_fail_regression_success':[k for k in expected if not allrows['flow'][k]['task_success'] and allrows['regression'][k]['task_success']], 'old_fail_new_success':[k for k in expected if allrows['flow'][k]['task_success'] and not allrows['control_flow'][k]['task_success']], 'old_success_new_fail':[k for k in expected if not allrows['flow'][k]['task_success'] and allrows['control_flow'][k]['task_success']], 'all_fail':[k for k in expected if all(not allrows[a][k]['task_success'] for a in paths)], 'all_success':[k for k in expected if all(allrows[a][k]['task_success'] for a in paths)]}
 selected=sorted(set(k for values in groups.values() for k in values[:2]));font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 for key in selected:
  canvas=Image.new('RGB',(1200,675),'white');draw=ImageDraw.Draw(canvas)
  for row,arm in enumerate(['regression','control_flow','flow']):
   e=allrows[arm][key];n=e['steps'];draw.text((6,row*225+4),f'{arm} task{key[0]}_init{key[1]} success={e["task_success"]} steps={n}; relative-time frames',font=font,fill='black')
   if arm=='flow':frames=frames_store[key]
   else:
    frames={};chosen={int(round(n*f)) for f in [0,.2,.4,.6,.8,1]}
    with imageio.get_reader(str(paths[arm]/e['pair_id']/'external.mp4'),ffmpeg_params=['-threads','1']) as reader:
     for j,frame in enumerate(reader):
      if j in chosen:frames[j]=Image.fromarray(frame)
   for col,(_,frame) in enumerate(sorted(frames.items())):canvas.paste(frame.resize((196,196)),(col*200,row*225+26))
  canvas.save(out/f'comparison_task{key[0]}_init{key[1]}.png')
 canvas=Image.new('RGB',(1600,900),'white');draw=ImageDraw.Draw(canvas)
 for j,key in enumerate(expected):
  x=j%10*160;y=j//10*180;e=allrows['flow'][key];canvas.paste(frames_store[key][e['steps']].resize((156,156)),(x,y+24));draw.text((x+2,y+2),f'{key[0]}/{key[1]} '+('OK' if e['task_success'] else 'FAIL'),font=font,fill='green' if e['task_success'] else 'red')
 canvas.save(out/'all50_final_frames.png')
 result=dict(status='PASS',episodes=50,metrics=metrics,paired_vs_regression=comparisons['regression'],paired_vs_control_flow=comparisons['control_flow'],per_task=tasks,all50_journals_and_queues_verified=True,all_noise_seeds_and_hashes_verified=noise_calls==metrics['flow']['calls'],noise_calls_verified=noise_calls,all_initial_states_and_images_matched=True,all50_videos_fully_decoded=True,decoded_frames=sum(x['frames'] for x in videos),videos=videos,representative_groups={k:[f'task{x[0]}_init{x[1]}' for x in v[:2]] for k,v in groups.items()},selected_timeline_pairs=[f'task{x[0]}_init{x[1]}' for x in selected],source_bindings_verified=len(p['input_sha256']),artifacts_sha256=artifacts,supervised_seconds=json.loads((RUN/'status.json').read_text())['elapsed_seconds'],elapsed_seconds=time.monotonic()-started,new_rollouts=0,new_training_updates=0,new_gpu_compute=0,limitations=['One training seed and adaptiveSpatial50;not final benchmark or unseen-task generalization.', 'Single-layer tail-consistent control shares fixed backbone,data,tail convention and50epochs; extra block increases parameter count; regression is a contextual reference.', 'Input hashes and physical journals checked with frozen runtime assertions;not a rerun of all50simulations or reconstruction of all raw wrist images from video.'])
 (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['videos','artifacts_sha256','per_task']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
