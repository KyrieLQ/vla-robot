#!/usr/bin/env python3
"""Durable finite stage; terminal event to the originating conversation only."""
import argparse
import hashlib
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
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


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
    assert plan.get('paired_live_amendment_approved') is True
    approval=json.loads((ROOT/plan['seed_approval_file']).read_text())
    assert approval['approved'] and hashlib.sha256((ROOT/plan['seed_approval_file']).read_bytes()).hexdigest()==plan['seed_approval_sha256']
    assert plan['resources']['paired_limit_seconds']==approval['change']['paired_cumulative_seconds_to']==16200
    assert plan['resources']['paired_limit_approved'] and plan['resources']['prior_paired_active_seconds']+plan['queue_timeout_seconds']<=16200
    assert plan['resources']['prior_recovery_active_seconds']+plan['queue_timeout_seconds']<=28800
    assert plan['seed_stability_round_prior_active_seconds']+plan['queue_timeout_seconds']<=2100
    assert json.loads((ROOT / plan['approval_file']).read_text())['approved'] is True
    thread = os.environ['CODEX_THREAD_ID']
    assert thread == plan['originating_thread']
    # Reuse the old whole-project exclusion lock; prevent duplicate VLA pipelines.
    lock = (ROOT / 'results/formal_active.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT / 'results' / a.name
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    state = {'status': 'RUNNING', 'name': a.name, 'phase': 'recovery_seed_stability', 'thread': thread,
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
            child = subprocess.Popen([str(ROOT / 'envs/openpi/bin/python'), '-u', str(ROOT / 'scripts/recovery_seed_pipeline_v1.py'),
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
                     experiment_finished=True, automatic_statistics_completed=(out / 'SEED_STABILITY_RESULTS.json').is_file())
        state.update(paired_cumulative_active_seconds=plan['resources']['prior_paired_active_seconds']+state['elapsed_seconds'],recovery_cumulative_active_seconds=plan['resources']['prior_recovery_active_seconds']+state['elapsed_seconds'],seed_study_cumulative_active_seconds=plan['seed_stability_round_prior_active_seconds']+state['elapsed_seconds'])
        write(out / 'status.json', state)
        registry = json.loads(registry_path.read_text())
        registry['runs'][a.name].update(status=state['status'], finished_at=state['finished_at'], experiment_finished=True, automatic_statistics_completed=state['automatic_statistics_completed'])
        write(registry_path, registry)
        event = {'thread': thread, 'run': str(out), 'status': state['status'], 'plan': str(plan_path),
                 'experiment_finished': True, 'automatic_statistics_completed': state['automatic_statistics_completed'],
                 'followup_processing': 'pending', 'created_at': now()}
        event_path = out / 'completion_event.json'
        write(event_path, event)
        message=(f'【VLA恢复标签种子重复性阶段完成/失败事件；仅一次通知】\n阶段 {a.name}，状态 {state["status"]}。\n'
                 f'运行目录：{out}\n事件：{event_path}\n冻结协议：{plan_path}\n'
                 '用户已明确批准配对累计有效窗口从4小时增至4.5小时，批准文件configs/recovery_seed_stability_20261014_APPROVED.json，不重复询问。原8小时恢复总量、最多两张允许GPU0–3、4GiB恢复/120GiB项目/100GiB保留不变；本轮总35分钟含失败尝试累计，新产物320MiB。'
                 '固定20fit场景每任务2个、三组新策略种子，continue_a与resample5共120分支，加10个同种子B对照，合计130；70条A/B物理重放、全部真实输入模型重放，动作1e-6/物理1e-7门槛不改。额外预检只复放已有成功/失败fit记录，不产生新闭环样本。'
                 '核对真实状态、完整130/130、全部输入/结果审核哈希、固定种子与场景、20x3收益矩阵、任务聚类描述性区间、A/B噪声、成本与累计4.5/8小时账，独立复算并更新记录/汇总。不得混用旧结果做本轮效果统计，不把三次结果当认证成功概率，不打开calibration/confirmation或自动训练。'
                 '若失败先保留已完成样本和冻结哈希，仅修复工程原因，在新记录中补必要部分；完整但待验收的闭环只补验收，禁止因结果重采、改样本或阈值。35分钟、4.5/8小时与存储预算均不重置；余量不足先准备可审阅方案。'
                 '至少8/20根收益随种子变化且A/B成败全一致只支持本子集标签对种子敏感，才准备另存期望收益数据协议；至多2根变化则不自动扩量，其余不确定，A/B不一致不得纯归因策略种子。完成独立分析和原范围内必要后继再标记processed。已有后继则去重，不恢复定时等卡、不停止其他任务。')
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
