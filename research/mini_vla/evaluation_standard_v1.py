"""Shared case selection. Metadata only; never starts a simulator or policy."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STANDARD='configs/mini_vla_evaluation_standard_20261019_v1.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_cases(level='development200',final_release=None):
    standard=json.loads((ROOT/STANDARD).read_text())
    path=ROOT/standard['case_manifest']
    if digest(path)!=standard['case_manifest_sha256']:
        raise ValueError('Evaluation manifest hash changed')
    manifest=json.loads(path.read_text());cases=manifest['cases']
    if level not in standard['levels']:
        raise ValueError('Unknown evaluation level')
    keys={(r['task_id'],r['initial_state_index']) for r in cases}
    if len(cases)!=500 or keys!={(t,i) for t in range(10) for i in range(50)}:
        raise ValueError('Invalid or duplicate Spatial cases')
    if standard['levels'][level].get('requires_final_release'):
        if final_release is None:
            raise PermissionError('Confirmation remains sealed until final method/checkpoints are frozen')
        release=json.loads(Path(final_release).read_text())
        if (release.get('status')!='FROZEN' or release.get('method_selection_closed') is not True
                or release.get('case_manifest_sha256')!=standard['case_manifest_sha256']
                or release.get('evaluation_standard_sha256')!=digest(ROOT/STANDARD)
                or not release.get('execution_protocol') or not release.get('checkpoints')):
            raise PermissionError('Incomplete final evaluation release')
        protocol=release['execution_protocol']
        protocol_path=(ROOT/protocol['path']).resolve()
        protocol_path.relative_to(ROOT.resolve())
        if digest(protocol_path)!=protocol['sha256']:
            raise ValueError('Final execution settings changed after freezing')
        for item in release['checkpoints']:
            checkpoint=(ROOT/item['path']).resolve()
            checkpoint.relative_to(ROOT.resolve())
            if digest(checkpoint)!=item['sha256']:
                raise ValueError('Final checkpoint changed after freezing')
    if level=='fast50':selected=[r for r in cases if r['fast_development']]
    elif level=='development200':selected=[r for r in cases if r['partition']=='development']
    elif level=='confirmation300':selected=[r for r in cases if r['partition']=='confirmation']
    else:selected=cases
    if len(selected)!=standard['levels'][level]['cases']:
        raise ValueError('Evaluation count mismatch')
    return selected
