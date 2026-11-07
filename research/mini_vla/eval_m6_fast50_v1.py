"""Evaluate the already-selected four-suite checkpoint on historicalSpatial50 only."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from .pipeline_v1 import ROOT,PY,write

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_fast50(evaluation):
    assert evaluation['suite']=='libero_spatial'
    assert evaluation['initial_state_indices']==[2,12,22,32,42]
    assert evaluation['task_ids']==list(range(10))
    assert evaluation['episodes_per_variant']==50
    assert evaluation['confirmation_access'] is False

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    pp=Path(a.plan).resolve();cfg=json.loads(pp.read_text());run=Path(a.run)
    for name,h in cfg['input_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in json.loads((run/'source_manifest.json').read_text()).items():assert sha(ROOT/name)==h,name
    validate_fast50(cfg['evaluation'])
    src=ROOT/cfg['trained_run'];accept=json.loads((ROOT/cfg['training_acceptance']).read_text());assert accept['status']=='PASS'
    assert sha(src/'training/best.pt')==accept['checkpoints']['best']['sha256']==cfg['checkpoint_sha256']
    view=run/'resnet_pretrained';view.mkdir();(view/'training').symlink_to(src/'training',target_is_directory=True)
    write(run/'reuse_manifest.json',dict(trained_run=str(src),selected_epoch=accept['selected_epoch'],checkpoint_sha256=cfg['checkpoint_sha256'],new_training_updates=0,new_eval_cases=50,confirmation_access=False))
    write(run/'pipeline.json',dict(status='RUNNING',stage='fixed50_development_eval'))
    command=[str(PY),str(ROOT/'scripts/gpu_run.py'),'--gpu',','.join(map(str,cfg['resources']['allowed_physical_gpus'])),'--gpus','1','--',str(PY),'-u','-B','-m','mini_vla.pipeline_m4_v2','--plan',str(pp),'--run',str(run),'--mode','evaluate','--variant','resnet_pretrained']
    with (run/'evaluation.log').open('x') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
    if not code:
        with (run/'analysis.log').open('x') as log:code=subprocess.run([str(PY),'-u','-B','-m','mini_vla.analyze_m6_fast50_v1','--plan',str(pp),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
    write(run/'pipeline.json',dict(status='COMPLETE_REVIEW_REQUIRED' if not code else 'WAITING_FOR_IDLE_GPU' if code==75 else 'FAILED',returncode=code));return code

if __name__=='__main__':raise SystemExit(main())
