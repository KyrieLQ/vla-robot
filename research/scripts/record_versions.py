#!/usr/bin/env python3
import datetime, importlib.metadata, json, os, pathlib, subprocess, sys

root=pathlib.Path(os.environ['PROJECT_ROOT'])
versions={'recorded_at':datetime.datetime.now().astimezone().isoformat(),
          'openpi_commit':subprocess.check_output(['git','-C',str(root/'repos/openpi'),'rev-parse','HEAD'],text=True).strip(),
          'libero_commit':subprocess.check_output(['git','-C',str(root/'repos/openpi/third_party/libero'),'rev-parse','HEAD'],text=True).strip(),
          'lerobot_commit':'0cf864870cf29f4738d3ade893e6fd13fbd7cdb5',
          'data_revision':json.loads((root/'reports/data_manifest.json').read_text())['source_revision'],
          'driver':'550.54.14','nvidia_smi_cuda_compatibility':'12.4','environments':{}}
probe='import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'
for name in ['openpi','libero']:
    py=root/'envs'/name/'bin/python';data=json.loads(subprocess.check_output([str(py),'-B','-c',probe],text=True))
    check=subprocess.run([str(root/'tools/bin/uv'),'pip','check','--python',str(py)],capture_output=True,text=True)
    data['dependency_check']='PASS' if check.returncode==0 else 'FAIL';data['dependency_check_output']=check.stdout+check.stderr
    versions['environments'][name]=data
    freeze=subprocess.check_output([str(root/'tools/bin/uv'),'pip','freeze','--python',str(py)],text=True)
    (root/'configs'/f'{name}_installed_freeze.txt').write_text(freeze)
(root/'reports/versions.json').write_text(json.dumps(versions,indent=2,sort_keys=True)+'\n')
lines=[f'{k}: {v}' for k,v in versions.items() if k!='environments']
for name,data in versions['environments'].items():
    lines+=['',f'[{name}]',data['python'],'dependency_check: '+data['dependency_check']]
    lines+=[f'{k}=={v}' for k,v in sorted(data['packages'].items())]
(root/'reports/versions.txt').write_text('\n'.join(lines)+'\n')
print('Recorded commits, exact package versions, and dependency checks for both environments.')
