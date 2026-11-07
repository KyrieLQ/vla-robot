import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];r=R/'results/mini_vla_m7_flow_depth2_dev50_20261104_run2';base=R/'results/mini_vla_m7_fixed_lora_flow_dev50_20261030_run1/flow/h5';rows=[]
for task in range(10):
 for init in [2,12,22,32,42]:
  pair=f'task{task}_init{init}';a=json.loads((base/pair/'episode.json').read_text());b=json.loads((r/'flow/h5'/pair/'episode.json').read_text());assert a['initial_image_sha256']==b['initial_image_sha256']
  for root,e in [(base,a),(r/'flow/h5',b)]:assert hashlib.sha256((root/pair/'trace.npz').read_bytes()).hexdigest()==e['trace_sha256']
  rows.append(int(b['task_success'])-int(a['task_success']))
x=np.array(rows);tasks=x.reshape(10,5).mean(1);ci=np.quantile(np.random.default_rng(20261019).choice(tasks,(10000,10)).mean(1),[.025,.975]).tolist();out=dict(episodes=50,control_successes=21,depth2_successes=28,wins=int((x==1).sum()),losses=int((x==-1).sum()),difference=float(x.mean()),task_cluster_descriptive_95_interval=ci,context_only=True,interventions='Both tail convention and depth differ from original21;not depth-only contrast');(r/'review/paired_original21_reference.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
q=dict(status='REVIEWED',all_videos_decoded=50,decoded_frames=7854,viewed_timeline_images=['comparison_task0_init22.png','comparison_task0_init12.png','comparison_task4_init2.png'],observations=['task0_init22: one-layer220 failure,depth2 75 success,regression71 success; depth2 reaches target placement in sampled frames.', 'task0_init12: one-layer84 success,depth2 220 failure; observed bowl contact and motion does not imply official success.', 'task4_init2: one-layer and regression220 failure,depth2 116 success; outcome advantage does not identify a unique mechanism.'],limitations='Six relative-time frames per row inspected; not a frame-by-frame semantic annotation. Success uses raw physical predicate, not image judgment.')
(r/'review/qualitative_video_review.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
