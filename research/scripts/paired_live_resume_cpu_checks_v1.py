"""Exercise real metadata races and prove non-metadata errors stay fatal."""
import datetime,json,os,pathlib,shutil,subprocess,tempfile,threading
from unittest import mock
from paired_live_audit_v1 import sha
from paired_live_resume_v1 import verify_entry
import storage_guard_v2 as storage
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def main():
    tests=[]
    def rejects(name,fn):
        try:fn()
        except (AssertionError,RuntimeError,OSError,ValueError):
            tests.append({'name':name,'status':'PASS_REJECTED'});return
        raise AssertionError('Unexpected acceptance: '+name)
    with tempfile.TemporaryDirectory(prefix='paired-resume-cpu-',dir=ROOT/'tmp') as tmp:
        tmp=pathlib.Path(tmp);race=tmp/'progress.json.tmp'
        transient=subprocess.CompletedProcess([],1,'1234\tpartial\n',"du: cannot access '%s': No such file or directory\n" % race)
        good=subprocess.CompletedProcess([],0,'4096\tcomplete\n','')
        with mock.patch.object(storage.subprocess,'run',side_effect=[transient,transient,good]) as run:
            assert storage.du_bytes(tmp)==4096 and run.call_count==3
        tests.append({'name':'transient_metadata_retried_and_partial_size_never_used','status':'PASS'})
        for label,stderr in [('permission_denied',"du: cannot read directory '%s': Permission denied\n" % tmp),
                             ('missing_actual_input',"du: cannot access '%s': No such file or directory\n" % (tmp/'policy_inputs.npz')),
                             ('outside_scope_metadata',"du: cannot access '/outside/progress.json.tmp': No such file or directory\n")]:
            failure=subprocess.CompletedProcess([],1,'1234\tpartial\n',stderr)
            with mock.patch.object(storage.subprocess,'run',return_value=failure) as run:
                rejects(label,lambda:storage.du_bytes(tmp));assert run.call_count==1
        with mock.patch.object(storage.subprocess,'run',return_value=transient) as run:
            rejects('retry_limit_is_finite',lambda:storage.du_bytes(tmp));assert run.call_count==5
        stop=threading.Event();errors=[]
        def writer():
            try:
                while not stop.is_set():
                    race.write_text('{"status":"RUNNING","payload":"'+('x'*8192)+'"}')
                    race.replace(tmp/'progress.json')
            except BaseException as exc:errors.append(str(exc))
        thread=threading.Thread(target=writer);thread.start()
        try:
            for _ in range(30):assert storage.du_bytes(tmp)>=0
        finally:stop.set();thread.join(timeout=10)
        assert not errors and not thread.is_alive()
        tests.append({'name':'30_real_du_reads_during_atomic_replace','status':'PASS'})
        plan=json.loads((ROOT/'configs/recovery_paired_live_20261014_v1.json').read_text())
        roots={r['root_id']:r for r in json.loads((ROOT/plan['root_manifest']).read_text())['roots']}
        complete=ROOT/'results/recovery_paired_live_20261014_v1/workers/worker0/roots/task0_init2_t60/continue_a'
        def spec(d,mode):
            row=json.loads((d/'result.json').read_text())
            return {'root_id':row['root_id'],'method':row['method'],'output':str(d),'mode':mode,
                    'files_sha256':{p.name:sha(p) for p in d.iterdir() if p.is_file() and not p.name.endswith('.tmp')}}
        entry=spec(complete,'verified');assert verify_entry(entry,roots[entry['root_id']],plan)['verification_status']=='PASS'
        tests.append({'name':'real_verified_branch_reuse_passes_all_bindings','status':'PASS'})
        source=ROOT/'results/recovery_paired_live_20261014_v1/workers/worker1/roots/task0_init28_t60/continue_a'
        pending=spec(source,'audit_only');assert verify_entry(pending,roots[pending['root_id']],plan)['verification_status']=='PENDING'
        tests.append({'name':'complete_unverified_rollout_classified_audit_only','status':'PASS'})
        copy=tmp/'pending-copy';shutil.copytree(source,copy)
        row=json.loads((copy/'result.json').read_text());row['output']=str(copy);(copy/'result.json').write_text(json.dumps(row))
        entry=spec(copy,'audit_only');verify_entry(entry,roots[entry['root_id']],plan)
        with (copy/'policy_inputs.npz').open('ab') as f:f.write(b'corruption')
        rejects('retained_input_hash_change',lambda:verify_entry(entry,roots[entry['root_id']],plan))
        shutil.copyfile(source/'policy_inputs.npz',copy/'policy_inputs.npz')
        row.update(status='TASK_SUCCESS',verification_status='PASS');(copy/'result.json').write_text(json.dumps(row))
        entry=spec(copy,'verified')
        rejects('unverified_rollout_cannot_masquerade_as_verified',lambda:verify_entry(entry,roots[entry['root_id']],plan))
    report={'status':'PASS','scope':'CPU-only disk-race and immutable resumption checks; no simulation or new scientific samples',
            'created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'tests':tests}
    (ROOT/'reports/paired_live_resume_cpu_checks_20261014.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
