"""Two-call diagnostic only: exact positive control and the rejected real image."""
import json,os,pathlib,socket,subprocess,time,urllib.request
import numpy as np
from PIL import Image
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def main():
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    out=ROOT/'reports/policy_image_effect_20261014';out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'configs/recovery_branch_plan_20261014_v4.json').read_text())
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=None;report={'scope':'Engineering sensitivity probe only; no simulated actions, no accepted branch outcome and no acceptance amendment.'}
    try:
        with (out/'server.log').open('x') as log:
            server=subprocess.Popen([str(ROOT/'envs/openpi/bin/python'),'-u',str(ROOT/'scripts/serve_recovery_branches_v1.py'),'--port',str(port),'--checkpoint',str(ROOT/plan['checkpoint'])],stdout=log,stderr=subprocess.STDOUT)
            for _ in range(150):
                if server.poll() is not None:raise RuntimeError('Probe model server exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:
                        if r.status==200:break
                except Exception:pass
                time.sleep(2)
            else:raise TimeoutError('Probe model startup timed out')
            client=WebsocketClientPolicy('127.0.0.1',port)
            specs=[('positive','task7_init44',60,ROOT/'results/recovery_branches_20261014_v3/workers/worker0/roots/task7_init44_t60/continue','root'),
                   ('rejected_actual_image','task7_init48',175,ROOT/'results/recovery_branches_20261014_v4/workers/worker1/roots/task7_init48_t60/continue','render_mismatch_175')]
            report['cases']=[]
            for label,pair,step,directory,prefix in specs:
                source=ROOT/plan['source_episodes']/pair;row=json.loads((source/'episode.json').read_text())
                with np.load(source/'policy_trace.npz') as data:state=data['robot_state'][step].copy();expected=data['action_chunk'][step//5].copy();original_image_hash=str(data['image_sha256'][step])
                images=[np.asarray(Image.open(directory/f'{prefix}_{cam}.png').convert('RGB')).copy() for cam in ('external','wrist')]
                import hashlib
                actual_hash=hashlib.sha256(images[0].tobytes()+images[1].tobytes()).hexdigest()
                assert (actual_hash==original_image_hash)==(label=='positive')
                result=client.infer({'observation/image':images[0],'observation/wrist_image':images[1],'observation/state':state,'prompt':row['language'],'_policy_seed':row['policy_seed'],'_policy_rng_advance':step//5})['actions']
                assert result.shape==expected.shape==(10,7) and np.isfinite(result).all()
                delta=np.abs(result-expected)
                item={'label':label,'pair_id':pair,'step':step,'actual_image_matches_original':actual_hash==original_image_hash,'max_action_chunk_difference':float(delta.max()),'mean_action_chunk_difference':float(delta.mean()),'original_action_tolerance':1e-6,'passes_original_action_gate':bool(np.all(delta<=1e-6)),'per_dimension_max':delta.max(axis=0).tolist()}
                report['cases'].append(item);np.savez_compressed(out/f'{label}.npz',prediction=result,reference=expected,robot_state=state)
                print(json.dumps(item),flush=True)
            assert report['cases'][0]['max_action_chunk_difference']==0,'Positive control must reproduce original output'
            report['status']='PASS_DIAGNOSTIC'
    except BaseException as exc:
        report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        report['server_cleaned_up']=True
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
