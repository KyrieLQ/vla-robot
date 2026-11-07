"""Prove cross-attempt hard links are counted once and errors remain fail-closed."""
import datetime,json,os,pathlib,subprocess,tempfile
from unittest import mock
from paired_live_audit_v1 import sha
import storage_guard_v3 as storage
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    tests=[]
    with tempfile.TemporaryDirectory(prefix='artifact-accounting-',dir=ROOT/'tmp') as tmp:
        root=pathlib.Path(tmp);a=root/'attempt_a';b=root/'attempt_b';a.mkdir();b.mkdir()
        source=a/'immutable.npz';source.write_bytes(bytes(range(256))*32);target=b/'immutable.npz'
        target.write_bytes(source.read_bytes());before=storage.du_paths_bytes([a,b]);digest=sha(source)
        temporary=b/'temporary.link';os.link(source,temporary);os.replace(temporary,target)
        after=storage.du_paths_bytes([a,b]);separate=storage.du_bytes(a)+storage.du_bytes(b)
        assert sha(source)==sha(target)==digest and source.stat().st_ino==target.stat().st_ino
        assert before-after==source.stat().st_blocks*512
        assert separate-after==source.stat().st_blocks*512
        assert storage.du_paths_bytes([a,b,a])==after
        tests.append({'name':'actual_cross_attempt_allocation_dedup_and_content_identity','status':'PASS','released_bytes':before-after,'independent_sum_overcount_bytes':separate-after})
        failure=subprocess.CompletedProcess([],1,'123\tpartial\n',"du: cannot access '%s': No such file or directory\n" % (b/'progress.json.tmp'))
        success=subprocess.CompletedProcess([],0,'456\ttotal\n','')
        with mock.patch.object(storage.subprocess,'run',side_effect=[failure,success]) as run:
            assert storage.du_paths_bytes([a,b])==456 and run.call_count==2
        tests.append({'name':'in_scope_metadata_race_retried_without_partial_count','status':'PASS'})
        for name,error in [('permission',"du: cannot read directory '%s': Permission denied\n" % a),
                           ('real_data_missing',"du: cannot access '%s': No such file or directory\n" % (b/'policy_inputs.npz'))]:
            with mock.patch.object(storage.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'1\ttotal\n',error)) as run:
                try:storage.du_paths_bytes([a,b])
                except storage.StorageMeasurementError:assert run.call_count==1
                else:raise AssertionError('Must not accept '+name)
            tests.append({'name':name,'status':'PASS_REJECTED'})
    report={'status':'PASS','created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'tests':tests,'scientific_inputs_or_thresholds_changed':False,'artifact_cap_gib':4,'concurrent_write_reserve_mib':128}
    (ROOT/'reports/artifact_accounting_cpu_checks_20261014.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
