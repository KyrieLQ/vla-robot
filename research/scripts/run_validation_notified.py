#!/usr/bin/env python3
"""A finite launch wrapper with one completion/failure notification, never periodic chat messages."""
import datetime, fcntl, json, os, pathlib, re, shutil, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    thread=os.environ['CODEX_THREAD_ID']
    lock=(ROOT/'results/notified_validation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    record=ROOT/'logs'/('validation_notification_'+stamp+'.json')
    data={'thread':thread,'wrapper_pid':os.getpid(),'status':'RUNNING','started_at':datetime.datetime.now().astimezone().isoformat(),
          'scope':'20 optimizer updates + resume 2 updates + 2 LIBERO policy episodes; no formal training',
          'periodic_chat_notifications':False,'authorization':'User explicitly requested starting now on 2026-10-13'}
    record.write_text(json.dumps(data,indent=2)+'\n')
    rc=subprocess.call(['bash',str(ROOT/'scripts/run_validation.sh'),'--gpu-only'])
    data.update(returncode=rc,status='COMPLETE' if rc==0 else ('WAITING_FOR_IDLE_GPU' if rc==75 else 'FAILED'),
                finished_at=datetime.datetime.now().astimezone().isoformat(),followup_processing='pending')
    latest=json.loads((ROOT/'reports/latest_pipeline.json').read_text())
    data['pipeline']=latest['path'];record.write_text(json.dumps(data,indent=2)+'\n')
    # This is a completion event for the newly authorized experiment, not the cancelled GPU watcher.
    message=('【VLA 部署验证完成/失败事件；仅一次通知】\n'
        f'本轮用户已于2026-10-13明确要求在空闲卡上开始原范围验证。状态：{data["status"]}。\n'
        f'运行目录：{latest["path"]}\n事件记录：{record}\n'
        '请读取实际日志和产物，去重后处理必要修复、20步真实训练、恢复2步与2个LIBERO闭环的验收，更新报告。'
        '不得自动开始正式长训练或完整benchmark；不得恢复已取消的定时等卡监控。'
        '如果当前会话已处理本次运行，只标记事件followup_processing=processed及处理时间，不重复发车。')
    try:
        cmd=shutil.which('codex')
        if not cmd:raise FileNotFoundError('codex executable unavailable')
        r=subprocess.run([cmd,'queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=30)
        data.update(delivery='accepted' if r.returncode==0 else 'failed',notification_returncode=r.returncode,
                    receipt_ids=re.findall(r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b',r.stdout))
    except subprocess.TimeoutExpired:data.update(delivery='unknown',notification_error='timeout; avoid duplicate resend')
    except OSError as exc:data.update(delivery='failed',notification_error=type(exc).__name__)
    record.write_text(json.dumps(data,indent=2)+'\n')
    return rc
if __name__=='__main__':raise SystemExit(main())
