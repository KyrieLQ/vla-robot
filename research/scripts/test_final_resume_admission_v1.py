"""No GPU: disk/time gates and immutable complete/private partial reuse contracts."""
import copy,hashlib,json,tempfile,time
from pathlib import Path
from mini_vla_resume_resource_waiter_v1 import eligibility
from mini_vla.final_resume_io_v1 import prepare_resume
R=Path(__file__).resolve().parents[1];start=time.monotonic()
plan=json.loads((R/'configs/mini_vla_final2000_20261021_v1.json').read_text());ledger=json.loads((R/'results/mini_vla_m6_20261019_ledger.json').read_text());allocated=214880624640
assert eligibility(plan,ledger,89*2**30,allocated)['status']=='WAITING_FOR_DISK'
assert eligibility(plan,ledger,115*2**30,allocated)['status']=='DISK_READY'
bad=copy.deepcopy(ledger);bad['modern']['stage_limits']['modern_final2000']=1
assert eligibility(plan,bad,115*2**30,allocated)['status']=='BUDGET_BLOCKED'
bad=copy.deepcopy(ledger);bad['project_cap_bytes']=1
assert eligibility(plan,bad,115*2**30,allocated)['status']=='PROJECT_STORAGE_BUDGET_BLOCKED'
with tempfile.TemporaryDirectory(dir=R/'tmp') as temp:
 tmp=Path(temp);source=tmp/'source';source.mkdir();dest=tmp/'dest';dest.mkdir()
 (source/'complete.txt').write_text('immutable\n');(source/'partial.txt').write_text('prefix\n');science=tmp/'science.json';science.write_text('{}')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 manifest=dict(status='AUDITED_PARTIAL_PRESERVE_OUTCOMES',source_run=str(source.relative_to(R)),checkpoint_sha256='test',scientific_plan_sha256=sha(science),complete_cases=1,partial_cases=1,cases=[dict(status='COMPLETE',files={'complete.txt':sha(source/'complete.txt')}),dict(status='PARTIAL',files={'partial.txt':sha(source/'partial.txt')})])
 m=tmp/'manifest.json';m.write_text(json.dumps(manifest));prepare_resume(dict(resume_manifest=str(m.relative_to(R)),checkpoint_sha256='test',scientific_plan=str(science.relative_to(R))),dest)
 assert (source/'complete.txt').stat().st_ino==(dest/'complete.txt').stat().st_ino
 assert (source/'partial.txt').stat().st_ino!=(dest/'partial.txt').stat().st_ino
 with (dest/'partial.txt').open('a') as f:f.write('suffix\n')
 assert (source/'partial.txt').read_text()=='prefix\n'
result=dict(status='PASS',disk_reserve_blocks_launch=True,budget_block_not_gpu_wait=True,project_cap_blocks_launch=True,complete_cases_reused_without_data_copy=True,partial_append_cannot_modify_original=True,elapsed_seconds=time.monotonic()-start,gpu_operations=0,new_simulation_episodes=0)
(R/'reports/final_resume_admission_cpu_contract_20261021.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
