"""One GPU admission; existing-input consistency, positive replay, fixed50 then analysis."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from .pipeline_v1 import write, ROOT, PY, LIBERO


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text());run=Path(a.run).resolve();started=time.monotonic()
    # The outer supervisor GPU gate selected this single device. Its process group
    # contains this pipeline, the policy server, and the simulation child.
    assert os.environ.get('VLA_GPU_SELECTION') and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    assert plan['evaluation']['episodes_per_variant']==50 and not plan['evaluation']['confirmation_access']
    def state(stage,**kwargs):write(run/'pipeline.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**kwargs))
    def launch(stage,command,timeout):
        state(stage)
        with (run/(stage+'.log')).open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    stage_limit=json.loads((run/'status.json').read_text())['timeout_seconds']
    def seconds_left():return stage_limit-(time.monotonic()-started)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=None
    try:
        state('online_contract')
        with (run/'server.log').open('a') as log:
            server=subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.policy_m8_v1','--plan',str(plan_path),'--port',str(port),'--contract-output',str(run/'online_contract.json')],stdout=log,stderr=subprocess.STDOUT)
        for _ in range(240):
            if server.poll() is not None:raise RuntimeError('Policy/consistency gate exited; see server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:%d/healthz'%port,timeout=1) as response:
                    if response.status==200:break
            except Exception:pass
            time.sleep(1)
        else:raise TimeoutError('Policy and consistency gate startup timeout')
        gate=json.loads((run/'online_contract.json').read_text());assert gate['status']=='PASS'
        base=json.loads((ROOT/plan['baseline_evaluation']/'result.json').read_text())
        baseline_steps=sum(e['steps'] for e in base['episodes'])
        # Conservative upper-duration sizing uses all220 steps/case and all44 calls/case,
        # not only successful/short episodes. No rollouts are generated for throughput.
        simulation_seconds=base['elapsed_seconds']/baseline_steps*11000*1.5
        infer_seconds=gate['inference_seconds_p95']*2200*1.3
        estimate=simulation_seconds+infer_seconds+120
        budget=dict(estimated_remaining_seconds=estimate,simulation_worstcase_seconds=simulation_seconds,
                    inference_worstcase_seconds=infer_seconds,remaining_arm_seconds=seconds_left(),
                    expected_policy_calls_max=2200,expected_env_steps_max=11000,
                    scope='Estimate only, including old RPC/video overhead in simulation term; runtime cap remains enforced')
        write(run/'throughput_gate.json',dict(status='PASS' if estimate<seconds_left()-30 else 'BUDGET_INSUFFICIENT',**budget))
        if estimate>=seconds_left()-30:return 78
        launch('positive_replay',[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_m8_v1','--plan',str(plan_path),'--output',str(run/'simulation_preflight'),'--variant','preflight'],120)
        launch('evaluation',[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_m8_v1','--plan',str(plan_path),'--output',str(run/'h5'),'--variant','h5','--port',str(port)],max(1,seconds_left()-30))
        launch('analysis',[str(PY),'-u','-B','-m','mini_vla.analyze_m8_dev50_v1','--plan',str(plan_path),'--run',str(run)],max(1,seconds_left()))
        write(run/'pipeline.json',dict(status='COMPLETE',stage=None,elapsed_seconds=time.monotonic()-started))
        return 0
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':raise SystemExit(main())
