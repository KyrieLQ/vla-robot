#!/usr/bin/env python3
"""Read-only, time-bounded GPU sampling and same-thread notifications; never launches training."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from gpu_run import idle, sample

ROOT = Path(__file__).resolve().parents[1]

def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def iso(unix=None):
    return datetime.datetime.fromtimestamp(time.time() if unix is None else unix).astimezone().isoformat()

def confirmed_idle(first, second):
    good = {g['uuid'] for g in first if idle(g)}
    return [g for g in second if g['uuid'] in good and idle(g)]

def eligible_alerts(gpus, sent_at, now, cooldown):
    return [g for g in gpus if now - sent_at.get(g['uuid'], float('-inf')) >= cooldown]

def notification_text(kind, event_path, config, gpus=None):
    purpose = {
        'idle': '检测到符合两次采样空闲条件的 GPU。先重新读取实际状态，再向用户汇报可用窗口。',
        'expired': '用户要求的 24 小时 GPU 定时检测已到期，检查进程已停止。读取监控摘要并汇报。',
        'error': 'GPU 查询连续失败。读取错误记录，核对监控状态并汇报；不能把查询失败视为没有 GPU。',
        'setup_test': '这是监控通知链路测试，不表示出现空闲 GPU。只核对状态并将本事件标记 processed；无需启动任何作业。',
    }[kind]
    return ('【VLA GPU 定时监控事件；仅监测通知，禁止自动训练】\n'
            '用户已明确选择每 5 分钟检测、持续 24 小时、发现空闲只通知本会话。\n'
            f'类型：{kind}\n事件文件：{event_path}\n监控配置：{config}\n'
            f'候选物理编号：{[g["index"] for g in (gpus or [])]}\n'
            f'{purpose}\n'
            '不要因先前实验授权自动启动 smoke/训练/评测，不停止其他任务、不读取其他用户命令行。'
            '将事件文件 processing 字段更新为 processed，并写入 processed_at；已处理的事件不重复处理。'
            '消息投递 accepted 不等于本会话已经处理；当前新用户指令优先。')

def deliver(event, config):
    # CLI is locally verified; no API key, remote destination or shell interpolation.
    event['delivery_attempts'] = event.get('delivery_attempts', 0) + 1
    event['attempted_unix'] = time.time()
    try:
        r = subprocess.run([config['codex_binary'], 'queue', '--thread', config['thread'],
                            '--message', event['message']], capture_output=True, text=True, timeout=30)
        # Store only a receipt UUID and status, not arbitrary configuration/errors/credentials.
        ids = re.findall(r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', r.stdout)
        event.update(delivery='accepted' if r.returncode == 0 else 'failed', returncode=r.returncode,
                     receipt_ids=ids, attempted_at=iso())
    except subprocess.TimeoutExpired:
        event.update(delivery='unknown', reason='CLI timeout; do not blindly resend', attempted_at=iso())
    except OSError as exc:
        event.update(delivery='failed', reason=type(exc).__name__, attempted_at=iso())
    return event

def make_event(directory, config_path, config, kind, sequence, gpus=None):
    key = f"{config['monitor_id']}:{kind}:{sequence}"
    event_id = hashlib.sha256(key.encode()).hexdigest()[:20]
    path = directory / 'events' / (event_id + '.json')
    if path.exists():
        return json.loads(path.read_text())
    event = dict(event_id=event_id, kind=kind, created_at=iso(), processing='pending',
                 thread=config['thread'], delivery='attempting', auto_training=False)
    event['message'] = notification_text(kind, path, config_path, gpus)
    atomic_json(path, event)
    event = deliver(event, config)
    atomic_json(path, event)
    return event

def summarize(history):
    by_gpu = {}
    valid = errors = 0
    if history.exists():
        for line in history.read_text().splitlines():
            row = json.loads(line)
            if row.get('error'):
                errors += 1
                continue
            valid += 1
            idle_ids = set(row['confirmed_idle_uuids'])
            for g in row['second']:
                k = str(g['index'])
                agg = by_gpu.setdefault(k, dict(samples=0, idle_samples=0, min_free_mib=g['free_mib'],
                                                max_free_mib=g['free_mib'], utilization_sum=0))
                agg['samples'] += 1
                agg['idle_samples'] += int(g['uuid'] in idle_ids)
                agg['min_free_mib'] = min(agg['min_free_mib'], g['free_mib'])
                agg['max_free_mib'] = max(agg['max_free_mib'], g['free_mib'])
                agg['utilization_sum'] += g['utilization']
    for agg in by_gpu.values():
        agg['mean_sampled_utilization'] = agg.pop('utilization_sum') / agg['samples']
    return dict(valid_checks=valid, query_errors=errors, per_gpu=by_gpu,
                interpretation='Instantaneous samples, not continuous occupancy or guaranteed exclusive availability')

def watch(config_path):
    config = json.loads(config_path.read_text())
    directory = config_path.parent
    (directory / 'events').mkdir(exist_ok=True)
    lock = (directory / 'watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A watcher for this configuration is already active; refusing duplicate')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    start = time.time()
    deadline = config['deadline_unix']
    if start >= deadline:
        raise SystemExit('Monitoring deadline already passed; no implicit renewal')
    state = dict(status='RUNNING', pid=os.getpid(), started_at=iso(), thread=config['thread'],
                 interval_seconds=config['interval_seconds'], deadline=iso(deadline),
                 checks=0, consecutive_errors=0, idle_event_count=0, auto_training=False,
                 next_check_at=iso(), last_check_at=None, last_confirmed_idle=[],
                 monitor_id=config['monitor_id'])
    next_check = time.monotonic()
    active_idle = set()
    announced_active = set()
    sent_at = {}
    error_sent = False
    atomic_json(directory / 'state.json', state)
    try:
        while not stop.is_set() and time.time() < deadline:
            if (directory / 'STOP').exists():
                stop.set()
                break
            if time.monotonic() >= next_check:
                state['checks'] += 1
                stamp = iso()
                try:
                    first = sample()
                    if stop.wait(config['confirmation_seconds']):
                        break
                    second = sample()
                    available = confirmed_idle(first, second)
                    current = {g['uuid'] for g in available}
                    row = dict(time=stamp, first=first, second=second,
                               confirmed_idle_uuids=sorted(current))
                    state.update(consecutive_errors=0, latest_gpus=second,
                                 last_confirmed_idle=[g['index'] for g in available])
                    # A notification deferred by the cooldown remains eligible
                    # on later checks while the card is still idle.
                    announced_active.intersection_update(current)
                    candidates = [g for g in available if g['uuid'] not in announced_active]
                    candidates = eligible_alerts(candidates, sent_at, time.time(), config['notification_cooldown_seconds'])
                    if candidates and state['idle_event_count'] < config['max_idle_events']:
                        event = make_event(directory, config_path, config, 'idle', state['checks'], candidates)
                        state['last_notification'] = {k: event.get(k) for k in ['event_id', 'delivery', 'receipt_ids']}
                        state['idle_event_count'] += 1
                        for g in candidates:
                            sent_at[g['uuid']] = time.time()
                            announced_active.add(g['uuid'])
                    active_idle = current
                except Exception as exc:
                    row = dict(time=stamp, error=type(exc).__name__)
                    state['consecutive_errors'] += 1
                    state['last_error'] = type(exc).__name__
                    if state['consecutive_errors'] >= 3 and not error_sent:
                        event = make_event(directory, config_path, config, 'error', state['checks'])
                        state['last_notification'] = {k: event.get(k) for k in ['event_id', 'delivery']}
                        error_sent = True
                with (directory / 'samples.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                state['last_check_at'] = stamp
                # Retry only definite delivery failures, at most three attempts.
                # Unknown/accepted receipts are never automatically resent.
                for path in (directory/'events').glob('*.json'):
                    event = json.loads(path.read_text())
                    if (event.get('delivery') == 'failed' and event.get('delivery_attempts', 0) < 3
                            and time.time() - event.get('attempted_unix', 0) >= config['interval_seconds']):
                        atomic_json(path, deliver(event, config))
                next_check += config['interval_seconds']
                if next_check < time.monotonic():
                    next_check = time.monotonic() + config['interval_seconds']
                state['next_check_at'] = iso(min(deadline, time.time() + max(0, next_check - time.monotonic())))
                atomic_json(directory / 'summary.json', summarize(directory / 'samples.jsonl'))
                print(json.dumps(dict(time=stamp, check=state['checks'], idle=state['last_confirmed_idle'],
                                      query_errors=state['consecutive_errors']), ensure_ascii=False), flush=True)
            state['heartbeat_at'] = iso()
            atomic_json(directory / 'state.json', state)
            stop.wait(min(30, max(0, next_check - time.monotonic()), max(0, deadline - time.time())))
        state.update(status='STOPPED' if stop.is_set() else 'EXPIRED', ended_at=iso(), next_check_at=None)
        atomic_json(directory / 'state.json', state)
        if state['status'] == 'EXPIRED':
            make_event(directory, config_path, config, 'expired', 0)
    except BaseException as exc:
        state.update(status='FAILED', ended_at=iso(), error=type(exc).__name__)
        atomic_json(directory / 'state.json', state)
        raise
    finally:
        atomic_json(directory / 'summary.json', summarize(directory / 'samples.jsonl'))
        lock.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['watch', 'test-notification', 'status', 'stop'])
    p.add_argument('--config', type=Path, default=ROOT/'monitoring/config.json')
    a = p.parse_args()
    if a.command == 'watch':
        watch(a.config)
    elif a.command == 'status':
        print((a.config.parent/'state.json').read_text())
    elif a.command == 'stop':
        (a.config.parent/'STOP').touch()
        print('Stop requested; the watcher will exit within its next <=30 second heartbeat.')
    else:
        config = json.loads(a.config.read_text())
        (a.config.parent/'events').mkdir(exist_ok=True)
        event = make_event(a.config.parent, a.config, config, 'setup_test', 0)
        print(json.dumps({k:event.get(k) for k in ['event_id','delivery','receipt_ids','returncode']}))

if __name__ == '__main__':
    main()
