"""Reuse all completed M3 work; execute only pretrained H5 evaluation and analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .pipeline_v1 import ROOT, PY, write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);p.add_argument('--mode',default='all');a=p.parse_args()
    spec=json.loads(Path(a.plan).read_text());run=Path(a.run);resume=spec['resume_only']
    source=(ROOT/resume['source_run']).resolve();source.relative_to(ROOT)
    original=(ROOT/resume['scientific_plan']).resolve();assert digest(original)==resume['scientific_plan_sha256']
    audit=ROOT/resume['acceptance'];assert digest(audit)==resume['acceptance_sha256']
    checked=json.loads(audit.read_text());assert checked['status']=='PARTIAL_ACCEPTED_FOR_EVAL_ONLY_RESUME'
    for name,h in checked['bound_files'].items():assert digest(ROOT/name)==h,name
    for name,h in json.loads((run/'source_manifest.json').read_text()).items():assert digest(ROOT/name)==h,name
    for variant in ['tiny_norm','resnet_random']:
        (run/variant).symlink_to(source/variant,target_is_directory=True)
    sub=run/'resnet_pretrained';sub.mkdir()
    (sub/'training').symlink_to(source/'resnet_pretrained/training',target_is_directory=True)
    assert not (source/'resnet_pretrained/h5/result.json').exists()
    write(run/'reuse_manifest.json',dict(source=str(source),acceptance_sha256=resume['acceptance_sha256'],
          reused_training_variants=3,reused_closed_loop_episodes=100,new_closed_loop_episodes=50,
          original_plan_sha256=resume['scientific_plan_sha256'],training_updates_added=0))
    write(run/'pipeline.json',dict(status='RUNNING',stage='eval_resnet_pretrained_only'))
    command=[str(PY),str(ROOT/'scripts/gpu_run.py'),'--gpu',','.join(map(str,spec['resources']['allowed_physical_gpus'])),'--gpus','1','--',
             str(PY),'-u','-B','-m','mini_vla.pipeline_m3_v1','--plan',str(original),'--run',str(run),'--mode','evaluate','--variant','resnet_pretrained']
    with (run/'eval_resnet_pretrained.log').open('x') as log:
        rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
    if rc:
        write(run/'pipeline.json',dict(status='WAITING_FOR_IDLE_GPU' if rc==75 else 'FAILED',returncode=rc));return rc
    write(run/'pipeline.json',dict(status='RUNNING',stage='analysis'))
    with (run/'analysis.log').open('x') as log:
        rc=subprocess.run([str(PY),'-u','-B','-m','mini_vla.analyze_m3_v1','--plan',str(original),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
    write(run/'pipeline.json',dict(status='COMPLETE' if rc==0 else 'FAILED',returncode=rc));return rc


if __name__=='__main__':raise SystemExit(main())
