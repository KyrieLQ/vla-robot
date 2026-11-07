"""Check the distributed source and recompute published metrics without imports."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "venv", "envs", "cache", "datasets", "__pycache__", "tmp", "repos"}


def main():
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not SKIP.intersection(p.relative_to(ROOT).parts)]
    python_files = json_files = 0
    for path in files:
        assert path.stat().st_size < 100 * 1024**2, f"Large Git object: {path.relative_to(ROOT)}"
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            python_files += 1
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            json_files += 1
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf8"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file(), f"Missing source: {entry['path']}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]

    metrics = json.loads((ROOT / "docs/libero_results.json").read_text(encoding="utf8"))
    with (ROOT / metrics["source"]).open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["suite"], r["task_id"], r["initial_state_index"]) for r in rows]
    assert len(rows) == len(set(keys)) == metrics["episodes"] == 2000
    assert all(r["success"] in {"0", "1"} for r in rows)
    successes = sum(int(r["success"]) for r in rows)
    assert successes == metrics["successes"] == 1565
    for suite, expected in metrics["suite_metrics"].items():
        part = [r for r in rows if r["suite"] == suite]
        assert len(part) == expected["episodes"]
        assert sum(int(r["success"]) for r in part) == expected["successes"]

    # Check entry-page links. Archived reports also refer to external run artifacts.
    for relative in ["README.md", "research/README.md", "robots/piper/README.md",
                     "docs/RESULTS.md", "THIRD_PARTY.md"]:
        page = ROOT / relative
        for url in re.findall(r"\]\(([^)\s]+)", page.read_text(encoding="utf8")):
            if "://" in url or url.startswith("#"):
                continue
            assert (page.parent / url.split("#")[0]).exists(), (relative, url)
    print(f"PASS: {python_files} Python files, {json_files} JSON files, {len(manifest['files'])} source hashes.")
    print(f"PASS: {successes}/{len(rows)} = {successes / len(rows):.2%}; all entry-page links resolve.")


if __name__ == "__main__":
    main()
