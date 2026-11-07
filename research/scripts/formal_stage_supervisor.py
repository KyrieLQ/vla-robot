#!/usr/bin/env python3
"""Finite durable stage with an event to its originating conversation; no idle polling."""
import argparse, datetime, fcntl, json, os, pathlib, re, shutil, signal, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--timeout',type=int,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    thread=os.environ['CODEX_THREAD_ID'];command=a.command[1:] if a.command[0]=='--' else a.command
    lock=(ROOT/'results/formal_active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'results'/a.name;out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    state={'status':'RUNNING','name':a.name,'thread':thread,'supervisor_pid':os.getpid(),'started_at':now(),'command':command,'timeout_seconds':a.timeout,'periodic_chat_notifications':False}
    path=out/'status.json';write(path,state)
    write(ROOT/'reports/latest_formal_stage.json',{'path':str(out),'thread':thread,'supervisor_pid':os.getpid()})
    registry_path=ROOT/'automation/formal_registry.json';registry_path.parent.mkdir(exist_ok=True)
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else {'runs':{}}
    registry['runs'][a.name]={'thread':thread,'path':str(out),'status':'RUNNING','supervisor_pid':os.getpid(),'started_at':state['started_at'],'event_path':str(out/'completion_event.json')}
    write(registry_path,registry)
    start=time.monotonic()
    with (out/'output.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=child.pid;write(path,state)
        try:code=child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            pipeline_path=out/'pipeline.json'
            if pipeline_path.exists():
                active=json.loads(pipeline_path.read_text()).get('active_pgid')
                if active:
                    try:os.killpg(active,signal.SIGINT)
                    except ProcessLookupError:pass
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            if pipeline_path.exists() and active:
                try:os.killpg(active,signal.SIGKILL)
                except ProcessLookupError:pass
            code=124
    state.update(status='COMPLETE' if code==0 else ('WAITING_FOR_IDLE_GPU' if code==75 else 'FAILED'),returncode=code,finished_at=now(),elapsed_seconds=time.monotonic()-start)
    write(path,state)
    registry=json.loads(registry_path.read_text());registry['runs'][a.name].update(status=state['status'],finished_at=state['finished_at']);write(registry_path,registry)
    event={'thread':thread,'run':str(out),'status':state['status'],'experiment_finished':True,'followup_processing':'pending','created_at':now()}
    event_path=out/'completion_event.json';write(event_path,event)
    message=(f'【VLA 正式训练阶段完成/失败事件；仅一次通知】\n用户于2026-10-13已明确要求开始正式训练。阶段 {a.name}，状态 {state["status"]}。\n运行目录：{out}\n事件记录：{event_path}\n'
      '读取 configs/formal_plan_20261013.json、实际状态和产物后去重处理。在已批准的双卡、30000步、spatial数据、配对评测和120GiB预算内继续必要阶段、修复、分析和报告。'
      '若本次阶段已经由本会话处理或后继任务在运行，仅标记processed和时间，不重复发车。不得恢复取消的定时等卡监控，不停止其他任务，不额外扩展训练步数。')
    try:
        result=subprocess.run([shutil.which('codex') or 'codex','queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=30)
        event.update(delivery='accepted' if result.returncode==0 else 'failed',notification_returncode=result.returncode,receipt_ids=re.findall(r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b',result.stdout))
    except subprocess.TimeoutExpired:event.update(delivery='unknown',notification_error='timeout; do not blindly resend')
    except OSError as exc:event.update(delivery='failed',notification_error=type(exc).__name__)
    write(event_path,event)
    return code
if __name__=='__main__':raise SystemExit(main())
