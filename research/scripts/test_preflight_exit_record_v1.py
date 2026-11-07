"""CPU subprocess cases: success, completed output then failure, incomplete failure."""
import json,subprocess,sys,tempfile
from pathlib import Path
from mini_vla.preflight_exit_record_v1 import record_exit
R=Path(__file__).resolve().parents[1]
def main():
    cases=[]
    with tempfile.TemporaryDirectory(dir=R/'tmp') as tmp:
        for name,complete,code in [('ok',True,0),('complete_nonzero',True,7),('incomplete',False,7)]:
            directory=Path(tmp)/name;directory.mkdir()
            if complete:
                (directory/'report.json').write_text(json.dumps(dict(status='PASS',formal_training_updates=0,new_simulation_episodes=0)))
                (directory/'progress.json').write_text(json.dumps(dict(stage='complete')))
                (directory/'metrics.json').write_text(json.dumps([dict(step=i) for i in range(1,11)]))
            child=subprocess.run([sys.executable,'-c','raise SystemExit('+str(code)+')'])
            report=record_exit(directory,child.returncode)
            assert report['returncode']==code and report['complete_worker_report_present']==complete and not report['automatic_gpu_retry_allowed']
            cases.append(dict(case=name,**report))
        signal=record_exit(Path(tmp)/'complete_nonzero',-15);assert signal['signal_number']==15
    result=dict(status='PASS',cases=cases,signal_recording=True,new_gpu_compute=0,new_optimizer_updates=0)
    out=R/'reports/pi05_preflight_exit_contract_20261105.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
