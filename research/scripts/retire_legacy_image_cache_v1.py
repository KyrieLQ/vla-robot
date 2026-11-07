"""Dry-run by default; only an explicitly approved exact two-file retirement may unlink."""
import argparse,fcntl,hashlib,json,os,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
R=Path(__file__).resolve().parents[1]
PROPOSAL=R/'configs/mini_vla_legacy_image_retirement_proposal_20261021_v1.json'
ALLOWED={'datasets/mini_vla_four_suite_128_20261019_v1/train/images.npy','datasets/mini_vla_four_suite_128_20261019_v1/validation/images.npy'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args();proposal=json.loads(PROPOSAL.read_text());assert {x['path'] for x in proposal['files']}==ALLOWED
 if not args.apply:
  print(json.dumps({'mode':'DRY_RUN_NO_DELETION','files':proposal['files'],'reclaim_gib':proposal['total_allocated_bytes']/2**30},indent=2));return
 auth=R/proposal['approval_required'];approval=json.loads(auth.read_text());assert approval['status']=='APPROVED' and approval['proposal_sha256']==sha(PROPOSAL)
 assert set(approval['approved_paths'])==ALLOWED
 assert approval['cache_stage_limit_seconds']==25200 and approval['modern_limit_seconds']==68400
 lock=(R/'results/formal_active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledgerpath=R/'results/mini_vla_m6_20261019_ledger.json';ledger=json.loads(ledgerpath.read_text());assert approval['thread']==json.loads((R/'configs/mini_vla_m8_m9_20261020_APPROVED.json').read_text())['thread']
 assert ledger['limit_seconds']==93600
 ledger['modern']['stage_limits']['modern_cache']=25200
 ledger['modern']['limit_seconds']=68400
 ledger['modern']['policy_revision_approval']={'path':str(auth.relative_to(R)),'sha256':sha(auth)}
 write(ledgerpath,ledger)
 source=json.loads((R/'results/mini_vla_m6_prepare_20261019_run3/data_integrity.json').read_text());base=R/'datasets/lerobot/physical-intelligence/libero'
 for e in source['files']:
  p=base/e['path'];assert p.is_file() and p.stat().st_size==e['bytes'],str(p)
 for part in ['train','validation']:
  for name in ['state','actions','episode_id','task_id']:
   assert (R/'datasets/mini_vla_four_suite_128_20261019_v1'/part/(name+'.npy')).is_file()
 auditpath=R/'reports/legacy_image_cache_retirement_20261021.json'
 audit=json.loads(auditpath.read_text()) if auditpath.exists() else dict(status='PREPARING',approval=str(auth.relative_to(R)),files={},attempts=[])
 start=time.monotonic();attempt={'started_at':now()};audit['attempts'].append(attempt);write(auditpath,audit)
 free_before=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
 try:
  for item in proposal['files']:
   name=item['path'];p=R/name
   previous=audit['files'].get(name)
   if p.exists():
    assert not p.is_symlink() and p.resolve()==R/name
    st=p.stat();assert st.st_size==item['bytes'] and st.st_nlink==1
    assert sha(p)==item['expected_sha256'],name
    record=dict(path=name,allocated_bytes=st.st_blocks*512,original_sha256=item['expected_sha256'],approval=str(auth.relative_to(R)),verified_at=now(),intent_recorded=True)
    audit['files'][name]=record;write(auditpath,audit)
    current=p.stat();assert (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)==(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    p.unlink()  # Exact allowlisted generated cache, after explicit approval and full hash.
   else:
    assert previous and previous['intent_recorded'],'Missing file without verified retirement intent'
    record=previous
   retired=ledger['modern'].setdefault('preexisting_asset_retirements',[])
   if not any(x['path']==name for x in retired):retired.append(record)
   write(ledgerpath,ledger)
   audit['files'][name].update(status='RETIRED',finished_at=now());write(auditpath,audit)
  audit['status']='COMPLETE'
  write(R/'datasets/mini_vla_four_suite_128_20261019_v1/retired_image_arrays.json',dict(status='IMAGE_ARRAYS_RETIRED_NUMERIC_ARRAYS_RETAINED',files=list(ALLOWED),approval=str(auth.relative_to(R)),retirement_record=str(auditpath.relative_to(R)),rebuild='Regenerate only image arrays from original pinned parquets in a new recorded run;historical complete.json retained as provenance, not current image availability'))
 finally:
  elapsed=time.monotonic()-start;attempt.update(elapsed_seconds=elapsed,finished_at=now());free_after=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
  audit.update(free_bytes_before=free_before,free_bytes_after=free_after,observed_free_delta_bytes=free_after-free_before)
  ledger['additional_entries'].append(dict(label='Approved_legacy_image_cache_retirement',stage_id='modern_cache',elapsed_seconds=elapsed,evidence=str(auditpath.relative_to(R))))
  ledger['total_recorded_seconds']=sum(x['elapsed_seconds'] for x in ledger['attempts']+ledger['additional_entries']);ledger['remaining_seconds']=ledger['limit_seconds']-ledger['total_recorded_seconds'];write(ledgerpath,ledger);write(auditpath,audit)
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
