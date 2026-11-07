#!/usr/bin/env python3
"""Finite parent-completion handoff. Never polls GPU availability or edits a live run."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,re,shutil,subprocess,time,traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def alive(pid):
    try:
        os.kill(int(pid),0)
        stat=pathlib.Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]=='Z':return False
        return True
    except ProcessLookupError:return False
    except FileNotFoundError:return False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);a=parser.parse_args()
    config_path=pathlib.Path(a.config);config=json.loads(config_path.read_text());thread=os.environ['CODEX_THREAD_ID'];assert thread==config['thread']
    out=ROOT/'automation'/config['name'];out.mkdir(exist_ok=True)
    lock=(out/'active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=out/'state.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('status') in ('SUCCESSOR_STARTED','NO_ADDITIONAL_EVALUATION'):
            print('Handoff already handled; no duplicate launch',flush=True);return 0
    parent=ROOT/config['parent_run'];approval=ROOT/config['approval_file'];successor=ROOT/'results'/config['successor_name']
    assert hashlib.sha256(approval.read_bytes()).hexdigest()==config['approval_sha256']
    assert json.loads(approval.read_text())['approved'] is True
    for source,digest in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest, 'Handoff source changed: '+source
    initial=json.loads((parent/'status.json').read_text())
    deadline=datetime.datetime.fromisoformat(initial['started_at']).timestamp()+initial['timeout_seconds']+300
    state={'status':'WAITING_FOR_PARENT','thread':thread,'pid':os.getpid(),'started_at':now(),'parent_run':str(parent),'successor_run':str(successor),
           'approval_file':str(approval),'config':str(config_path),'persistent_gpu_polling':False,'periodic_chat_notifications':False}
    write(path,state)
    registry_path=ROOT/'automation/research_registry.json'
    registry=json.loads(registry_path.read_text());registry.setdefault('handoffs',{})[config['name']]={'state':str(path),'pid':os.getpid(),'thread':thread,'parent_run':str(parent),'successor_run':str(successor),'status':'WAITING_FOR_PARENT'}
    registry['runs'][parent.name].update(approved_suffix_handoff=str(path),successor_reserved=config['successor_name'])
    write(registry_path,registry)
    def finalize_registry():
        registry=json.loads(registry_path.read_text());registry['handoffs'][config['name']].update(status=state['status'],updated_at=now());write(registry_path,registry)
    def notify_failure(error):
        event={'status':'HANDOFF_FAILED','created_at':now(),'thread':thread,'parent_run':str(parent),'error':error,'followup_processing':'pending'}
        event_path=out/'completion_event.json';write(event_path,event)
        message=(f'【VLA 已批准后缀修订衔接失败；仅一次通知】\n衔接状态：{path}\n事件：{event_path}\n父任务：{parent}\n'
                 '用户已批准所有未消费前缀/后缀图像（含终止记录帧）的诊断例外，实际策略输入/根图像/物理动作结果门槛仍保留。读取真实日志、注册信息及批准文件后修复；不得重复请求同一批准，不重复启动已在运行的后继，不扩大样本/预算，不恢复等卡监控。')
        try:
            result=subprocess.run([shutil.which('codex') or 'codex','queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=30)
            event.update(delivery='accepted' if result.returncode==0 else 'failed',receipt_ids=re.findall(r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b',result.stdout))
        except subprocess.TimeoutExpired:event.update(delivery='unknown',notification_error='Do not blindly resend after timeout')
        except OSError as exc:event.update(delivery='failed',notification_error=type(exc).__name__)
        write(event_path,event)
    try:
        dead_samples=0
        while True:
            current=json.loads((parent/'status.json').read_text())
            state.update(parent_status=current['status'],heartbeat_at=now());write(path,state)
            if current['status']!='RUNNING':break
            if not alive(current['supervisor_pid']):
                dead_samples+=1
                if dead_samples>=2:raise RuntimeError('Parent status is stale and its recorded supervisor has exited')
            else:dead_samples=0
            if time.time()>deadline:raise TimeoutError('Parent exceeded its finite supervision deadline')
            time.sleep(10)
        if current['status'] not in ('INCOMPLETE_RENDER_REVIEW','COMPLETE'):
            # The parent already sends its own terminal event. Do not blindly retry
            # consumed-image, root-image, numeric or unrelated software failures.
            state.update(status='PARENT_FAILURE_REQUIRES_REVIEW',finished_at=now(),parent_status=current['status']);write(path,state);finalize_registry();return 1
        state['status']='WAITING_FOR_PARENT_CLEANUP';write(path,state)
        limit=time.monotonic()+90
        while alive(current['supervisor_pid']):
            if time.monotonic()>limit:raise TimeoutError('Parent terminal event/lock cleanup has not completed')
            time.sleep(2)
        if (successor/'status.json').exists():
            existing=json.loads((successor/'status.json').read_text());assert existing['thread']==thread
            state.update(status='SUCCESSOR_STARTED',adopted_existing_successor=True,finished_at=now());write(path,state);finalize_registry();return 0
        from prepare_recovery_branch_v4 import prepare
        for source,digest in config['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest, 'Handoff source changed: '+source
        state['status']='FREEZING_SUCCESSOR';write(path,state)
        prepared=prepare(parent,approval,ROOT/config['successor_plan'])
        state['prepared']=prepared
        if prepared['next_action']=='NO_ADDITIONAL_EVALUATION':
            state.update(status='NO_ADDITIONAL_EVALUATION',finished_at=now());write(path,state);finalize_registry();return 0
        # The normal project lock and finite admission run again in the successor.
        command=[str(ROOT/'envs/openpi/bin/python'),'-u',str(ROOT/'scripts/recovery_branch_supervisor_v4.py'),
                 '--name',config['successor_name'],'--plan',prepared['plan']]
        with (ROOT/'logs'/(config['successor_name']+'.supervisor.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state['successor_supervisor_pid']=child.pid;state['status']='STARTING_SUCCESSOR';write(path,state)
        for _ in range(60):
            if (successor/'status.json').exists():break
            if child.poll() is not None:raise RuntimeError('Successor exited before writing its registered status')
            time.sleep(1)
        assert (successor/'status.json').exists(),'Successor did not register within 60 seconds'
        state.update(status='SUCCESSOR_STARTED',finished_at=now());write(path,state);finalize_registry()
        event_path=parent/'completion_event.json'
        if event_path.exists():
            event=json.loads(event_path.read_text());event.update(followup_processing='processed',processed_at=now(),successor=str(successor),
                resolution='Automatic finite handoff applied the explicit suffix approval, audited and reused all complete branches, froze remaining work and registered the successor. Full research interpretation follows the combined complete results.')
            write(event_path,event)
        registry=json.loads(registry_path.read_text());registry['runs'][parent.name].update(followup_processing='processed',processed_at=now(),successor=config['successor_name']);write(registry_path,registry)
        print(json.dumps({'status':'SUCCESSOR_STARTED','run':str(successor),'retained':prepared['retained_complete_branches'],'missing':prepared['missing_branches']}),flush=True)
        return 0
    except BaseException as exc:
        state.update(status='HANDOFF_FAILED',finished_at=now(),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());write(path,state);finalize_registry();notify_failure(type(exc).__name__+': '+str(exc));raise
if __name__=='__main__':raise SystemExit(main())
