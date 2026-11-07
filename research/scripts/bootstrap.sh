#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu GIT_LFS_SKIP_SMUDGE=1
  cd "$PROJECT_ROOT"
  if [[ ! -d repos/openpi/.git ]]; then
    git clone https://github.com/Physical-Intelligence/openpi.git repos/openpi
    git -C repos/openpi checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
  fi
  [[ "$(git -C repos/openpi rev-parse HEAD)" == 215abfb217dbac7d5f1273282331b9b1866c0479 ]] || {
    echo 'Unexpected openpi commit; refusing to reset or overwrite the checkout' >&2; return 2;
  }
  if [[ -s reports/openpi_local.patch ]]; then
    if ! git -C repos/openpi apply --reverse --check "$PROJECT_ROOT/reports/openpi_local.patch" 2>/dev/null; then
      git -C repos/openpi apply --check "$PROJECT_ROOT/reports/openpi_local.patch"
      git -C repos/openpi apply "$PROJECT_ROOT/reports/openpi_local.patch"
    fi
  fi
  /usr/bin/python3 -B - <<'PY'
import json, urllib.request, pathlib, zipfile, os
root=pathlib.Path(os.environ['PROJECT_ROOT'])
meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/uv/0.8.22/json',timeout=30))
item=next(x for x in meta['urls'] if x['filename'].endswith('manylinux_2_17_x86_64.manylinux2014_x86_64.whl'))
print('uv bootstrap download bytes',item['size'],flush=True)
dest=root/'cache'/item['filename']
if not dest.exists():
 with urllib.request.urlopen(item['url'],timeout=60) as src, dest.open('wb') as out:
  while True:
   chunk=src.read(1024*1024)
   if not chunk:break
   out.write(chunk)
import hashlib
assert dest.stat().st_size==item['size']
assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['digests']['sha256']
with zipfile.ZipFile(dest) as z:
 name=next(n for n in z.namelist() if n.endswith('/uv'))
 out=root/'tools/bin/uv';out.write_bytes(z.read(name));out.chmod(0o755)
PY
  uv --version
  cd "$PROJECT_ROOT/repos/openpi"
  git submodule update --init --depth 1 third_party/libero
  uv python install 3.11.13 3.8.20
  uv sync --frozen --no-dev --python 3.11.13
}
main "$@"
