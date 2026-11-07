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
    plan_path = pathlib.Path(a.plan)
    plan = json.loads(plan_path.read_text())
    assert plan.get('prefix_replay_acceptance_amendment_approved') is True, 'Unapproved engineering amendment; do not launch'
    assert plan.get('unconsumed_suffix_image_amendment_approved') is True, 'Suffix scope must be explicitly approved'
    thread = os.environ['CODEX_THREAD_ID']
    assert thread == plan['originating_thread']
    # Reuse the old whole-project exclusion lock; prevent duplicate VLA pipelines.
    lock = (ROOT / 'results/formal_active.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT / 'results' / a.name
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    state = {'status': 'RUNNING', 'name': a.name, 'phase': 'recovery_branch_feasibility', 'thread': thread,
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
            child = subprocess.Popen([str(ROOT / 'envs/openpi/bin/python'), '-u', str(ROOT / 'scripts/recovery_branch_pipeline_v4.py'),
                                      '--output', str(out), '--plan', str(plan_path)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = child.pid
            write(out / 'status.json', state)
            code = child.wait(timeout=plan['queue_timeout_seconds'])
    except subprocess.TimeoutExpired:
        code = 124
        state['error'] = 'Finite 8-hour branch queue time cap reached'
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
        message = (f'【VLA 恢复分支阶段完成/失败事件；仅一次通知】\n用户已明确要求开始 1–2 个月项目改进，利用空闲 GPU。阶段 {a.name}，状态 {state["status"]}。\n'
                   f'运行目录：{out}\n事件记录：{event_path}\n冻结方案：{plan_path}\n'
                   '用户已于2026-10-14分别明确批准历史前缀与继续后缀（含终止记录帧）中未消费图像只作差异记录；分支起点和实际推理输入图像仍逐像素相同，物理状态/动作/预测块/最终结果保持原门槛。两项批准均沿用，不重复请求同一许可。已完成分支按冻结哈希复用，保留累计资源记账。'
                   '若覆盖不足，不训练模块、不发布完整成功率；实际推理输入或数值违例不在上述例外中。核对所有continue轨迹的真实调用步图像、动作和物理量；已批准未消费图像例外不能扩大为修改模型输入或成功判据。'
                   '核对真实状态、预检和全量开发集产物后去重处理；必要修复保留已完成样本，不改样本或阈值。完成配对统计、失败归因边界、项目记录和汇总；据证据冻结并推进原研究目标内下一阶段。'
                   '本队列为 200 个开发场景的 269 个根状态、1076 个配对分支，无新增训练更新。核对所有继续分支逐步复现和 60/120 步分层统计；离线上界不当作上线选择器效果。至少 8 个不同基线失败场景可救回且复现全过后，按冻结 fit/calibration 划分准备进展/收益模块；先另存模型、损失、种子和预算协议。否则先改进恢复候选，不因负结果重抽样。确认集 300 场景方法定型前不得开启。'
                   '继续遵守同时最多两张允许的 GPU 0–3、120GiB 总量与100GiB文件系统保留；不恢复已取消的定时等卡监控，不停止其他任务。'
                   '若已处理或后继任务已运行，不重复发车，只在必要工作完成后标记 followup_processing=processed 和时间。消息 accepted 不等于分析已处理。')
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
