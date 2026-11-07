#!/usr/bin/env python3
"""Finite detached stage supervisor; no GPU polling or persistent daemon."""
import argparse, datetime, json, os, pathlib, subprocess, time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('name');p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]
    # Bound CPU use of the child and all its worker threads/processes.
    cpu_affinity=sorted(os.sched_getaffinity(0))[:8]
    os.sched_setaffinity(0,cpu_affinity)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=root/'logs'/f'{a.name}_{stamp}';run.mkdir()
    state={'name':a.name,'supervisor_pid':os.getpid(),'thread':os.environ.get('CODEX_THREAD_ID'),
           'started_at':datetime.datetime.now().astimezone().isoformat(),'command':a.command,'state':'RUNNING',
           'cpu_affinity':cpu_affinity}
    def save():
        (run/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    start=time.monotonic()
    with (run/'output.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=child.pid;save()
        try:rc=child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            # This group was created by this supervisor; never target another job.
            import signal
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            rc=124
    state.update(returncode=rc,state='PASS' if rc==0 else ('WAITING_FOR_IDLE_GPU' if rc==75 else 'FAIL'),elapsed_seconds=time.monotonic()-start,
                 ended_at=datetime.datetime.now().astimezone().isoformat())
    save()
    (run/'completion_event.json').write_text(json.dumps({'stage_finished':True,'followup_processed':False,
                                                       'state':state['state'],'thread':state['thread']},indent=2)+'\n')
    return rc
if __name__=='__main__':raise SystemExit(main())
