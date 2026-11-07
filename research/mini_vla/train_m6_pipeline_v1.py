"""Approved four-suite M4 training only; never launches confirmation rollouts."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from .pipeline_v1 import ROOT,PY,write

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def data_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text());run=Path(a.run);started=time.monotonic()
    for path,h in plan['input_sha256'].items():assert sha(ROOT/path)==h,path
    acceptance=json.loads((ROOT/plan['data_acceptance']).read_text());assert acceptance['status']=='PASS'
    for path,h in acceptance['cache_all_files_sha256'].items():assert data_sha(ROOT/path)==h,path
    complete=json.loads((ROOT/plan['cache']/'complete.json').read_text());assert complete['binding']['split_sha256']==sha(ROOT/plan['split']) and complete['binding']['preprocessing_sha256']==sha(ROOT/plan['preprocessing'])
    assert complete['parts']['train']['frames']==243000 and complete['parts']['validation']['frames']==30465
    def launch(stage,command,timeout):
        for path,h in json.loads((run/'source_manifest.json').read_text()).items():assert sha(ROOT/path)==h,path
        write(run/'pipeline.json',dict(status='RUNNING',stage=stage,pipeline_started_monotonic=started))
        with (run/(stage+'.log')).open('x') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
        if code:raise subprocess.CalledProcessError(code,command)
    def gate(command):return [str(PY),str(ROOT/'scripts/gpu_run.py'),'--gpu',','.join(map(str,plan['resources']['allowed_physical_gpus'])),'--gpus','1','--']+command
    try:
        launch('throughput',gate([str(PY),'-u','-B','-m','mini_vla.preflight_m6_training_v1','--plan',str(plan_path),'--run',str(run)]),180)
        time.sleep(5)
        launch('training',gate([str(PY),'-u','-B','-m','mini_vla.experiment_m4_v1','train','--plan',str(plan_path),'--run',str(run),'--variant','resnet_pretrained']),int(plan['resources']['per_attempt_seconds']))
        result=json.loads((run/'training/report.json').read_text());assert result['epochs']==50 and result['updates']==94950
        assert result['selected_validation']['frames']==30465 and result['selected_validation']['valid_target_steps']==296235
        assert sha(run/'training/best.pt')==result['checkpoint_sha256']['best']
        write(run/'pipeline.json',dict(status='COMPLETE_REVIEW_REQUIRED',stage=None,training_updates=94950,confirmation_accessed=False,new_simulation_episodes=0))
        return 0
    except subprocess.CalledProcessError as exc:
        write(run/'pipeline.json',dict(status='WAITING_FOR_IDLE_GPU' if exc.returncode==75 else 'BUDGET_INSUFFICIENT' if exc.returncode==78 else 'FAILED',returncode=exc.returncode));return exc.returncode

if __name__=='__main__':raise SystemExit(main())
