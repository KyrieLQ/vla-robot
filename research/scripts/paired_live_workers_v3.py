#!/usr/bin/env python3
"""Two already-admitted physical devices; paired roots are disjoint across workers."""
import argparse,datetime,json,os,pathlib,signal,subprocess,time
ROOT=pathlib.Path(os.environ['PROJECT_ROOT']);SCRIPTS=pathlib.Path(__file__).resolve().parent
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    run=pathlib.Path(a.run);plan=json.loads(pathlib.Path(a.plan).read_text());selection=json.loads(pathlib.Path(os.environ['VLA_GPU_SELECTION']).read_text());assert len(selection['selected'])==plan['workers']
    (run/'workers').mkdir();(run/'selections').mkdir();children=[];logs=[];gate_open=False
    cpus=sorted(os.sched_getaffinity(0))
    try:
        for worker,gpu in enumerate(selection['selected']):
            gpu=dict(gpu,logical_index=0);record=dict(selection,selected=[gpu],worker=worker,parent_admission=os.environ['VLA_GPU_SELECTION'])
            sp=run/'selections'/f'worker{worker}.json';write(sp,record)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['VLA_GPU_SELECTION']=str(sp)
            log=(run/f'worker{worker}.log').open('x');logs.append(log)
            child=subprocess.Popen([str(ROOT/'envs/openpi/bin/python'),'-u',str(SCRIPTS/'paired_live_worker_v3.py'),'--run',str(run),'--plan',a.plan,'--worker',str(worker)],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            assigned=cpus[worker::plan['workers']];os.sched_setaffinity(child.pid,assigned or cpus)
        write(run/'workers.json',{'pids':[p.pid for p in children],'selected_physical_gpus':[g['index'] for g in selection['selected']],'count':plan['workers'],'same_episode_in_one_worker':True})
        while True:
            codes=[p.poll() for p in children]
            if any(code not in (None,0) for code in codes):raise RuntimeError('Branch worker failed: '+str(codes))
            if not gate_open:
                files=[run/'workers'/f'worker{w}'/'preflight.json' for w in range(plan['workers'])]
                if all(p.is_file() for p in files):
                    checks=[json.loads(p.read_text()) for p in files];assert all(c['status']=='PASS' for c in checks)
                    write(run/'COLLECTION_GATE.json',{'status':'PASS','created_at':datetime.datetime.now().astimezone().isoformat(),'workers_passed':plan['workers'],'preflight_branches_passed':sum(len(c['rows']) for c in checks),'all_workers':checks});gate_open=True
            if all(code==0 for code in codes):break
            time.sleep(2)
        assert gate_open
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=40)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
if __name__=='__main__':raise SystemExit(main())
