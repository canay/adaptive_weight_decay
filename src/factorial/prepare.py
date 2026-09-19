"""Copy verified predecessor inputs; seal a new non-destructive run package."""
import json, shutil, sys
from pathlib import Path
from runner import ROOT, atomic, digest, now
PROJECT=ROOT.parents[1]
OLD=PROJECT/'experiments/2026-08-26_codex_local_strong_baseline_confirmatory/outputs/main'

def prepare():
    cfg=json.loads((ROOT/'config.json').read_text())
    manifest=json.loads((OLD/'dataset_cache/manifest.json').read_text())
    (ROOT/'inputs').mkdir(exist_ok=True)
    for row in manifest['tasks']:
        if row['task'].endswith('_full'):
            src=OLD/'dataset_cache'/row['file']
            assert digest(src)==row['sha256'], row['file']
            shutil.copy2(src,ROOT/'inputs'/row['file'])
    raw=OLD/'confirmatory_raw.csv'
    assert digest(raw)=='535DF250D5513A5D16C8816C9FAA1AD53A0B293C0055266EB935C5BA9E41D953'
    shutil.copy2(raw,ROOT/'inputs/previous_raw.csv')
    assert not (ROOT/'freeze.json').exists(), 'REFUSE_REFREEZE'
    shutil.copy2(PROJECT/'MD/02_design/F06_MC007_PROTOCOL.md',ROOT/'PROTOCOL.md')
    files=[p for p in ROOT.rglob('*') if p.is_file() and
           (p.relative_to(ROOT).parts[0] in ('src','inputs') or
            (p.parent==ROOT and p.name in ('config.json','PROTOCOL.md','BASE_PROTOCOL.md')))
           and '__pycache__' not in p.parts and p.suffix!='.pyc']
    total=sum(p.stat().st_size for p in files)
    assert total<cfg['run_output_budget_bytes']//2
    atomic(ROOT/'freeze.json',dict(schema_version=1,run_id=cfg['run_id'],created_utc=now(),
           visibility='MC007_MAIN_NOT_GENERATED; MC006_R2_2520_RESULTS_SEEN_AND_HELD_FOR_SCALING_DEFECT; E0_ALREADY_SEEN',
           files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)},
           bytes=total,planned_main_units=2520))
    print(json.dumps(dict(status='PASS',bytes=total,files=len(files),freeze_sha256=digest(ROOT/'freeze.json'))))

if __name__=='__main__':prepare()
