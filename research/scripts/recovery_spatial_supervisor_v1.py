#!/usr/bin/env python3
"""Durable finite stage; terminal event to the originating conversation only."""
import argparse
import datetime
import fcntl
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import time
import traceback

ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])


def now():
    return datetime.datetime.now().astimezone().isoformat()


def write(p, d):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--plan', required=True)
    a = p.parse_args()
    plan_path = pathlib.Path(a.plan).resolve()
    plan = json.loads(plan_path.read_text())
    acceptance=json.loads((ROOT / plan['parent_run'] / 'review/acceptance.json').read_text())
    assert acceptance['status']=='PASS' and acceptance['robust_rescuable_original_failures']>=8
    plan['queue_timeout_seconds']=plan['resources']['max_active_seconds']
    thread = os.environ['CODEX_THREAD_ID']
    assert thread == plan['originating_thread']
    # Reuse the old whole-project exclusion lock; prevent duplicate VLA pipelines.
    lock = (ROOT / 'results/formal_active.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT / 'results' / a.name
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    state = {'status': 'RUNNING', 'name': a.name, 'phase': 'spatial_progress_representation_fit_pilot', 'thread': thread,
             'supervisor_pid': os.getpid(), 'started_at': now(), 'plan': str(plan_path),
             'timeout_seconds': plan['queue_timeout_seconds'], 'periodic_chat_notifications': False,
             'experiment_finished': False, 'analysis_review_processed': False}
    write(out / 'status.json', state)
    write(ROOT / 'reports/latest_research_stage.json', {'path': str(out), 'thread': thread, 'supervisor_pid': os.getpid()})
    registry_path = ROOT / 'automation/research_registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'runs': {}}
    registry['runs'][a.name] = dict(state, path=str(out), event_path=str(out / 'completion_event.json'))
    write(registry_path, registry)
    start = time.monotonic()
    child = None
    code = 1
    try:
        with (out / 'output.log').open('x') as log:
            child = subprocess.Popen([str(ROOT / 'envs/openpi/bin/python'), '-u', str(ROOT / 'scripts/recovery_spatial_pipeline_v1.py'),
                                      '--output', str(out), '--plan', str(plan_path)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = child.pid
            write(out / 'status.json', state)
            code = child.wait(timeout=plan['queue_timeout_seconds'])
    except subprocess.TimeoutExpired:
        code = 124
        state['error'] = 'Frozen value-pilot active-time cap reached'
    except BaseException as exc:
        state.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        # All pipeline children deliberately inherit this known, newly-created group.
        # This never targets other users or earlier experiments.
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            # Clean any remaining descendants after the controller exits.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        state.update(status='COMPLETE' if code == 0 else ('INCOMPLETE_RENDER_REVIEW' if code == 78 else ('GPU_UNAVAILABLE' if code == 75 else 'FAILED')),
                     returncode=code, finished_at=now(), elapsed_seconds=time.monotonic()-start,
                     experiment_finished=True, automatic_statistics_completed=(out / 'SPATIAL_RESULTS.json').is_file())
        write(out / 'status.json', state)
        registry = json.loads(registry_path.read_text())
        registry['runs'][a.name].update(status=state['status'], finished_at=state['finished_at'], experiment_finished=True, automatic_statistics_completed=state['automatic_statistics_completed'])
        write(registry_path, registry)
        event = {'thread': thread, 'run': str(out), 'status': state['status'], 'plan': str(plan_path),
                 'experiment_finished': True, 'automatic_statistics_completed': state['automatic_statistics_completed'],
                 'followup_processing': 'pending', 'created_at': now()}
        event_path = out / 'completion_event.json'
        write(event_path, event)
        message=(f'【VLA空间进展表示fit试验完成/失败事件；仅一次通知】\n阶段 {a.name}，状态 {state["status"]}。\n'
                 f'运行目录：{out}\n事件：{event_path}\n冻结协议：{plan_path}\n'
                 '原动作条件收益试验负结果保留。本轮仅140fit场景186根、1763个已存真实调用及46个原120根输入，合计1809次引用；无新仿真、VLA更新或calibration/confirmation输入。'
                 '核对全部原始输入/物理时刻/来源哈希，186根池化特征须与历史完全一致；网格和全局均值两类几何回归各3固定投影种子，再接正则化收益模型，五折全episode隔离。真实几何只作训练监督及明确非部署对照，不能泄漏到学习模型输入。'
                 '独立复算30几何模型/35critic、预测、各折标准化、联合NLL/收益MSE和几何误差；核对全部fit样本。OOF是自适应开发证据不是独立检验。'
                 '仅空间组合的NLL与收益MSE都低于任务时间先验和匹配全局模型，且至少2/3种子MSE低于对应全局，才准备另存校准/缓存/成本协议。原零误伤与20%成本/种子稳定门槛不降低；否则保留阴性结果，结合oracle对照界定表征/标签限制，不继续盲目调参。'
                 '最多1允许GPU0–3、CPU4线程、900秒、128MiB新增产物；总4GiB/120GiB/100GiB与累计4/8小时预算不重置，失败续跑也累计。先审核、记录汇总和预算内必要后继，再标记processed。已有后继则去重；不恢复定时等卡、不停止其他任务。')
        try:
            result = subprocess.run([shutil.which('codex') or 'codex', 'queue', '--thread', thread, '--message', message],
                                    capture_output=True, text=True, timeout=30)
            event.update(delivery='accepted' if result.returncode == 0 else 'failed', notification_returncode=result.returncode,
                         receipt_ids=re.findall(r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', result.stdout))
        except subprocess.TimeoutExpired:
            event.update(delivery='unknown', notification_error='Timeout; do not blindly resend')
        except OSError as exc:
            event.update(delivery='failed', notification_error=type(exc).__name__)
        write(event_path, event)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
