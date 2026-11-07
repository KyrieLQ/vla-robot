"""Deduplicate immutable byte-identical NPZ/PNG artifacts, preserving every path."""
import argparse,datetime,fcntl,filecmp,json,os,pathlib,stat,subprocess,time
from paired_live_audit_v1 import sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def within(path,root):
    try:path.resolve().relative_to(root.resolve());return True
    except ValueError:return False
def allocated(paths):
    result=subprocess.run(['du','-s','-c','-B1',*[str(p) for p in paths]],capture_output=True,text=True,check=True,env=dict(os.environ,LC_ALL='C'))
    fields=result.stdout.splitlines()[-1].split();assert fields[1]=='total';return int(fields[0])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',required=True);parser.add_argument('--apply',action='store_true');a=parser.parse_args()
    inventory_path=pathlib.Path(a.inventory).resolve();inventory=json.loads(inventory_path.read_text())
    parent=inventory_path.parent;paths=[pathlib.Path(p) for p in inventory['stage_paths']]
    assert parent==ROOT/'results/recovery_paired_live_20261014_v2'
    assert all(within(p,ROOT) for p in paths)
    lock=(ROOT/'results/formal_active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for d in paths:
        if (d/'status.json').exists():assert json.loads((d/'status.json').read_text())['status']!='RUNNING'
    if not a.apply:
        print(json.dumps({'status':'DRY_RUN','duplicate_groups':len(inventory['duplicate_groups'])}));return
    before=allocated(paths);assert before+16*2**20<4*2**30,'No headroom for a durable deduplication ledger'
    start=time.monotonic();changed=0;skipped=0;groups=0
    log_path=parent/'deduplication_operations.jsonl'
    with log_path.open('x') as ledger:
        ledger.write(json.dumps({'event':'BEGIN','created_at':now(),'inventory_sha256':sha(inventory_path),'allocated_before':before,'scope':'Only regular immutable NPZ/PNG, identical full bytes, same filesystem/ownership/mode. Every pathname retained.'})+'\n');ledger.flush();os.fsync(ledger.fileno())
        for group in inventory['duplicate_groups']:
            buckets={}
            for item in group['files']:
                path=pathlib.Path(item['path']);assert any(within(path,d) for d in paths)
                assert path.suffix in ('.png','.npz') and not path.is_symlink()
                st=path.stat();assert stat.S_ISREG(st.st_mode)
                assert st.st_size==group['size'] and sha(path)==group['sha256']
                key=(st.st_dev,st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode))
                buckets.setdefault(key,[]).append(path)
            for candidates in buckets.values():
                canonical=candidates[0];source_stat=canonical.stat();groups+=1
                for path in candidates[1:]:
                    original=path.stat()
                    if (original.st_dev,original.st_ino)==(source_stat.st_dev,source_stat.st_ino):skipped+=1;continue
                    assert sha(canonical)==sha(path)==group['sha256']
                    assert filecmp.cmp(canonical,path,shallow=False),'Full bytes differ despite hash grouping'
                    temporary=path.with_name(path.name+'.dedup.tmp');assert not temporary.exists()
                    record={'event':'PREPARE','path':str(path),'canonical':str(canonical),'sha256':group['sha256'],
                            'prior_inode':original.st_ino,'prior_mtime_ns':original.st_mtime_ns,'prior_mode':stat.S_IMODE(original.st_mode),
                            'prior_allocated_bytes':original.st_blocks*512,'size':original.st_size}
                    ledger.write(json.dumps(record)+'\n');ledger.flush()
                    os.link(canonical,temporary);os.replace(temporary,path)
                    assert sha(path)==group['sha256'] and path.stat().st_ino==canonical.stat().st_ino
                    ledger.write(json.dumps({'event':'COMMIT','path':str(path),'sha256':group['sha256'],'new_inode':path.stat().st_ino})+'\n');ledger.flush()
                    changed+=1
                    if changed%100==0:os.fsync(ledger.fileno())
        ledger.write(json.dumps({'event':'END','finished_at':now(),'changed_paths':changed,'already_linked_paths':skipped})+'\n');ledger.flush();os.fsync(ledger.fileno())
    after=allocated(paths)
    result={'status':'COMPLETE','finished_at':now(),'elapsed_seconds':time.monotonic()-start,'changed_paths':changed,
            'already_linked_paths':skipped,'allocated_bytes_before':before,'allocated_bytes_after':after,
            'net_allocated_bytes_released':before-after,'all_pathnames_retained':True,'full_byte_comparison_before_replacement':True,
            'all_content_hashes_unchanged':True,'json_logs_and_verdicts_not_replaced':True,
            'scope':'Only immutable completed-run NPZ/PNG; source inode times per pathname recorded in ledger. No science inputs, outcomes or gates changed.',
            'ledger':str(log_path),'ledger_sha256':sha(log_path)}
    (parent/'deduplication_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
