"""Exercise both real interpreter environments without creating a sim or CUDA model."""
import argparse,ast,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-only',action='store_true');args=ap.parse_args();start=time.monotonic()
 if not args.cpu_only:assert os.environ.get('VLA_GPU_SELECTION'),'Run through idle GPU admission for EGL import'
 env=os.environ.copy();env.update(PYTHONPATH=str(R/'tools/m8_processor_deps_v1')+':'+str(R)+':'+str(R/'scripts')+':'+str(R/'repos/openpi/third_party/libero')+':'+env.get('PYTHONPATH',''),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 code_libero="""
import importlib.util,json,os
from pathlib import Path
from egl_device import select_egl
mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy,imageio
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from mini_vla import libero_eval_paired_v1,paired_eval_journal_v1
p=Path('repos/openpi/examples/libero/main.py').resolve()
spec=importlib.util.spec_from_file_location('libero_helper',p)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
assert suite.n_tasks==10
print(json.dumps({'libero_import':True,'evaluator_and_helper_import':True,'suite_tasks':suite.n_tasks,'environment_created':False}))
"""
 code_policy="""
import json
from transformers import AutoModelForImageTextToText
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from mini_vla import policy_paired_v1,pipeline_paired_dev50_v2,analyze_paired_dev50_v1
print(json.dumps({'transformers_import':True,'policy_server_pipeline_analysis_import':True,'cuda_model_created':False}))
"""
 if args.cpu_only:
  code_libero="import importlib.util,json;from libero.libero import benchmark; spec=importlib.util.find_spec('libero.libero.envs');assert spec is not None; assert benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0).n_tasks==10;print(json.dumps({'libero_benchmark_import':True,'env_module_resolvable':True,'EGL_import_deferred_until_idle_GPU':True}))"
 results={}
 for name,code in [('libero',code_libero),('openpi',code_policy)]:
  child=subprocess.run([str(R/'envs'/name/'bin/python'),'-B','-c',code],cwd=R,env=env,capture_output=True,text=True,timeout=120)
  results[name]=dict(returncode=child.returncode,stdout=child.stdout,stderr=child.stderr)
  if child.returncode:
   failure=dict(status='FAILED',interpreters=results,elapsed_seconds=time.monotonic()-start)
   (R/'reports/paired_eval_environment_v2_failure_20261029.json').write_text(json.dumps(failure,indent=2)+'\n')
   print(json.dumps(results));raise RuntimeError(name+' import smoke failed')
 source=(R/'scripts/mini_vla_paired_eval_supervisor_v2.py').read_text();ast.parse(source);assert "+str(R/'repos/openpi/third_party/libero')+':'+env.get('PYTHONPATH','')" in source
 result=dict(status='PASS',interpreters=results,elapsed_seconds=time.monotonic()-start,new_training_updates=0,new_simulation_episodes=0,cuda_models_loaded=0,gpu_admission=os.environ.get('VLA_GPU_SELECTION','CPU_ONLY'),scope='CPU module resolution and policy imports; EGL imports deferred' if args.cpu_only else 'Actual selectedGPU EGL import and dualPython module imports; no model or environment instance')
 (R/('reports/paired_eval_environment_v2_cpu_contract_20261029.json' if args.cpu_only else 'reports/paired_eval_environment_v2_contract_20261029.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
