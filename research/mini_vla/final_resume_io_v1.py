"""Reuse immutable complete cases; copy incomplete journals privately before continuation."""
import hashlib,json,os,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def prepare_resume(plan,run):
 manifest=json.loads((R/plan['resume_manifest']).read_text());assert manifest['status']=='AUDITED_PARTIAL_PRESERVE_OUTCOMES'
 assert manifest['checkpoint_sha256']==plan['checkpoint_sha256']
 assert manifest['scientific_plan_sha256']==sha(R/plan['scientific_plan'])
 source_root=(R/manifest['source_run']).resolve();log=[]
 for case in manifest['cases']:
  for rel,h in case['files'].items():
   source=(source_root/rel).resolve();source.relative_to(source_root);assert sha(source)==h
   dest=run/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
   if case['status']=='COMPLETE':os.link(source,dest)
   else:shutil.copy2(source,dest)
   assert sha(dest)==h
   log.append(dict(path=rel,sha256=h,mode='immutable_hardlink' if case['status']=='COMPLETE' else 'private_partial_copy'))
 (run/'resume_reuse_manifest.json').write_text(json.dumps(dict(source_run=str(source_root),files=log,complete_cases=manifest['complete_cases'],partial_cases=manifest['partial_cases']),indent=2)+'\n')
