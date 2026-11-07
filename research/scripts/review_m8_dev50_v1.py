"""Independent raw-file audit of fixed50, including complete video decoding."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_m8_dev50_20261021_run1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True);plan=json.loads((R/'configs/mini_vla_m8_dev50_20261021_v1.json').read_text());recorded=json.loads((RUN/'analysis.json').read_text())
 for f,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/f)==h,f
 for f,h in plan['input_sha256'].items():assert sha(R/f)==h,f
 assert json.loads((R/plan['training_acceptance']).read_text())['status']=='PASS'
 gate=json.loads((RUN/'online_contract.json').read_text());assert gate['status']=='PASS' and gate['samples']==120 and gate['all_features_bitwise_equal'] and gate['maximum_action_error']==0
 positive=json.loads((RUN/'simulation_preflight/result.json').read_text());assert positive['status']=='PASS' and positive['success_verified'] and positive['max_qpos_error']==0
 stats=json.loads((R/plan['preprocessing']).read_text())['normalization']['actions'];m=np.array(stats['mean'],dtype=np.float32);s=np.array(stats['scale'],dtype=np.float32)
 cases=[(t,i) for t in range(10) for i in (2,12,22,32,42)];rows=[];all_latency=[];all_server=[];movies=[];review_frames=[];all_final=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 for task,init in cases:
  pair=f'task{task}_init{init}';d=RUN/'h5'/pair;b=R/plan['baseline_evaluation']/pair;e=json.loads((d/'episode.json').read_text());base=json.loads((b/'episode.json').read_text())
  assert e['task_id']==task and e['initial_state_index']==init and e['plan_sha256']==sha(R/'configs/mini_vla_m8_dev50_20261021_v1.json') and e['checkpoint_sha256']==plan['checkpoint_sha256']
  assert sha(d/'trace.npz')==e['trace_sha256'] and sha(d/'external.mp4')==e['video_sha256'];assert sha(b/'trace.npz')==base['trace_sha256']
  with np.load(d/'trace.npz') as z:tr={k:z[k] for k in z.files}
  with np.load(b/'trace.npz') as z:
   for k in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(tr[k],z[k],atol=1e-7,rtol=0)
  assert e['initial_image_sha256']==base['initial_image_sha256'] and e['language']==base['language']
  n=e['steps'];calls=(n+4)//5;assert 1<=n<=220 and len(tr['actions'])==n and e['policy_calls']==calls
  assert e['task_success']==bool(tr['done'][-1]) and not tr['done'][:-1].any() and (e['task_success'] or n==220)
  np.testing.assert_array_equal(tr['call_steps'],np.arange(0,n,5))
  for step in range(n):np.testing.assert_array_equal(tr['actions'][step],tr['action_chunks'][step//5,step%5])
  np.testing.assert_allclose(tr['action_chunks'],np.clip(tr['normalized_action_chunks']*s+m,-1,1),atol=1e-6,rtol=0)
  for k in ['actions','robot_state','normalized_action_chunks','latency','server_ms']:assert np.isfinite(tr[k]).all()
  for k in ['observation_sha256','vlm_input_sha256']:assert len(tr[k])==calls and all(len(x)==64 for x in tr[k])
  all_latency.extend(tr['latency'][1:]);all_server.extend(tr['server_ms'][1:]);selected={0,n//4,n//2,3*n//4,n};frames={};count=0
  with imageio.get_reader(str(d/'external.mp4'),ffmpeg_params=['-threads','1']) as reader:
   for j,frame in enumerate(reader):
    assert frame.shape==(256,256,3)
    if j in selected:frames[j]=Image.fromarray(frame)
    count+=1
  assert count==n+1,(pair,count,n)
  movies.append(dict(pair_id=pair,frames=count,sha256=e['video_sha256']))
  all_final.append((pair,e['task_success'],frames[n]))
  if not e['task_success'] or (e['task_success'] and not base['task_success']):review_frames.append((pair,e['task_success'],base['task_success'],frames))
  rows.append(dict(task_id=task,initial_state_index=init,m8=int(e['task_success']),m6=int(base['task_success']),steps=n,calls=calls))
  print('verified',pair,'video',count,flush=True)
 tasks=[dict(task_id=t,m6=sum(x['m6'] for x in rows if x['task_id']==t),m8=sum(x['m8'] for x in rows if x['task_id']==t),episodes=5) for t in range(10)]
 dif=np.array([x['m8']-x['m6'] for x in rows]);taskdiff=np.array([(x['m8']-x['m6'])/5 for x in tasks]);indices=np.random.default_rng(20261019).integers(0,10,size=(10000,10));interval=np.quantile(taskdiff[indices].mean(1),[.025,.975]).tolist()
 result=dict(status='PASS',episodes=50,successes=int(sum(x['m8'] for x in rows)),baseline_successes=int(sum(x['m6'] for x in rows)),wins=int((dif==1).sum()),losses=int((dif==-1).sum()),difference=float(dif.mean()),task_cluster_descriptive_95_interval=interval,
  initial_physics_and_images_all_matched=True,all50_queues_verified=True,all50_videos_fully_decoded=True,decoded_video_frames=sum(x['frames'] for x in movies),policy_calls=sum(x['calls'] for x in rows),steps=sum(x['steps'] for x in rows),rpc_median_ms=float(np.median(all_latency)*1000),rpc_p95_ms=float(np.quantile(all_latency,.95)*1000),server_median_ms=float(np.median(all_server)),per_task=tasks,rows=rows,videos=movies,elapsed_seconds=time.monotonic()-start,new_simulation_episodes=0,new_optimizer_updates=0,confirmation_accessed=False)
 assert result['successes']==recorded['metrics']['m8']['successes'] and result['wins']==recorded['paired']['wins'] and result['losses']==recorded['paired']['losses'];np.testing.assert_allclose(interval,recorded['paired']['task_cluster_descriptive_95_interval'],atol=1e-12,rtol=0)
 for page in range((len(review_frames)+5)//6):
  subset=review_frames[page*6:page*6+6];canvas=Image.new('RGB',(1000,len(subset)*220),'white');draw=ImageDraw.Draw(canvas)
  for r,(pair,success,bs,frames) in enumerate(subset):
   draw.text((5,r*220+2),f'{pair} M8={success} M6={bs}; start / quarter / half / three-quarter / final',font=font,fill='black')
   for c,(step,im) in enumerate(sorted(frames.items())):canvas.paste(im.resize((196,196)),(c*200,r*220+24))
  canvas.save(out/f'failure_and_rescue_timeline_{page+1}.png')
 canvas=Image.new('RGB',(1600,5*180),'white');draw=ImageDraw.Draw(canvas)
 for j,(pair,success,im) in enumerate(all_final):
  x=(j%10)*160;y=(j//10)*180;canvas.paste(im.resize((156,156)),(x,y+24));draw.text((x+2,y+2),pair+(' OK' if success else ' FAIL'),fill='green' if success else 'red',font=font)
 canvas.save(out/'all50_final_frames.png')
 result['elapsed_seconds']=time.monotonic()-start;(out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['rows','videos','per_task']})
if __name__=='__main__':main()
