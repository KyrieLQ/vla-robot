"""Persist process-exit evidence without equating exit status with artifact completeness."""
import hashlib,json
from pathlib import Path

def record_exit(directory,returncode):
    directory=Path(directory);files={};complete=False
    for name in ['report.json','progress.json','metrics.json']:
        path=directory/name
        if path.exists():files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        report=json.loads((directory/'report.json').read_text());progress=json.loads((directory/'progress.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
        complete=(report['status']=='PASS' and progress['stage']=='complete' and len(metrics)==10 and [row['step'] for row in metrics]==list(range(1,11)) and report['formal_training_updates']==report['new_simulation_episodes']==0)
    except (OSError,ValueError,KeyError,TypeError):pass
    result=dict(returncode=int(returncode),signal_number=-returncode if returncode<0 else None,
        complete_worker_report_present=complete,artifact_sha256=files,
        handling='INDEPENDENT_ARTIFACT_REVIEW_BEFORE_ANY_RETRY' if returncode and complete else 'INCOMPLETE_PRESERVE_FOR_REPAIR' if returncode else 'VERIFY_NORMAL_COMPLETION',
        automatic_gpu_retry_allowed=False)
    temp=directory/'worker_exit.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(directory/'worker_exit.json')
    return result
