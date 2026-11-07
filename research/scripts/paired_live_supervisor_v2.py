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
    assert plan.get('paired_live_amendment_approved') is True, 'Unapproved engineering amendment; do not launch'
    assert json.loads((ROOT / plan['approval_file']).read_text())['approved'] is True
    thread = os.environ['CODEX_THREAD_ID']
    assert thread == plan['originating_thread']
    # Reuse the old whole-project exclusion lock; prevent duplicate VLA pipelines.
    lock = (ROOT / 'results/formal_active.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT / 'results' / a.name
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    state = {'status': 'RUNNING', 'name': a.name, 'phase': 'recovery_paired_live', 'thread': thread,
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
            child = subprocess.Popen([str(ROOT / 'envs/openpi/bin/python'), '-u', str(ROOT / 'scripts/paired_live_pipeline_v2.py'),
                                      '--output', str(out), '--plan', str(plan_path)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = child.pid
            write(out / 'status.json', state)
            code = child.wait(timeout=plan['queue_timeout_seconds'])
    except subprocess.TimeoutExpired:
        code = 124
        state['error'] = 'Approved paired-round or cumulative recovery time cap reached'
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
                     experiment_finished=True, automatic_statistics_completed=(out / 'ANALYSIS.json').is_file())
        write(out / 'status.json', state)
        registry = json.loads(registry_path.read_text())
        registry['runs'][a.name].update(status=state['status'], finished_at=state['finished_at'], experiment_finished=True, automatic_statistics_completed=state['automatic_statistics_completed'])
        write(registry_path, registry)
        event = {'thread': thread, 'run': str(out), 'status': state['status'], 'plan': str(plan_path),
                 'experiment_finished': True, 'automatic_statistics_completed': state['automatic_statistics_completed'],
                 'followup_processing': 'pending', 'created_at': now()}
        event_path = out / 'completion_event.json'
        write(event_path, event)
        message = (f'【VLA 统一配对恢复阶段完成/失败事件；仅一次通知】\n阶段 {a.name}，状态 {state["status"]}。\n'
                   f'运行目录：{out}\n事件记录：{event_path}\n冻结协议：{plan_path}\n'
                   '用户已明确批准统一配对新协议，批准文件configs/recovery_paired_live_20261014_APPROVED.json；不得重复请求已获许可。'
                   '同一200开发场景/269根状态，五组共1345分支：continue_a主对照、同种子continue_b噪声对照和原三候选。'
                   '保存全部真实无损推理输入，所有分支独立输入重放动作容差1e-6，538个A/B继续分支独立动作重放物理容差1e-7和结果精确核验；不再要求实时后缀逐帧等于历史轨迹。'
                   '旧严格轮次结果完整保留，不得混入新配对轮次效果统计。同一已批准配对轮次的完成分支按冻结哈希复用，完整但中断验收的闭环只补独立验收，不重采结果；累计4/8小时预算不重置。核对1345/1345、全部审核哈希、60/120分层、任务聚类区间、重复对照噪声及成本。'
                   '至少8个不同原基线失败场景在A/B均失败而某候选成功且审核全过后，按原fit/calibration划分冻结并准备收益模块；否则按原目标改进候选，不能因负结果改阈值或重抽样。'
                   '完成独立结果验收、失败归因边界、实验记录/汇总和原范围内后继；离线上界不当上线选择器效果。确认集300场景方法定型前不得开启。'
                   '继续遵守最多两张允许GPU0–3、项目120GiB、文件系统保留100GiB、恢复阶段产物4GiB、累计8小时有效窗口及本轮4小时窗口；保留累计资源记账。'
                   '不恢复取消的定时等卡监控、不停止其他任务、不重复发车。事件accepted不等于分析处理。必要处理结束后标记followup_processing=processed和时间。')
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
