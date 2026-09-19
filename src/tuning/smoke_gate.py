"""Determinism, genuine checkpoint resume, liveness and bounded ETA admission."""
from core import *
from engine import validate_payload
from selection import lock_selection,evaluation

HB_FIELDS={'timestamp','run_id','unit_id','attempt_id','pid','phase','phase_started_at',
    'unit_elapsed_seconds','completed_atomic_units','planned_atomic_units','last_durable_checkpoint_at',
    'process_tree_cpu_seconds','wrapper_cpu_seconds','process_tree_pids'}

def advancing_pair(events):
    for a,b in zip(events,events[1:]):
        assert HB_FIELDS<=a.keys() and HB_FIELDS<=b.keys()
        if (all(a[k]==b[k] for k in ['run_id','unit_id','attempt_id','pid']) and
            a['phase']==b['phase']=='training' and b['timestamp']>a['timestamp'] and
            b['unit_elapsed_seconds']>a['unit_elapsed_seconds'] and
            b['process_tree_cpu_seconds']>a['process_tree_cpu_seconds'] and
            b['inner_completed']>a['inner_completed']):
            return [a,b]
    return None

def main():
    cfg=load(ROOT/'config.json');frozen=verify_freeze();environment=env();checks={};signatures=[]
    for phase in ['smoke','repeat']:
        out=ROOT/'outputs'/phase;term=load(out/'terminal_status.json')
        assert term['status']=='COMPLETED' and term['exit_code']==0 and term['completed_units']==68
        sh,sel=lock_selection(out,cfg,frozen,environment,True);sig={}
        for c in development(cfg,True)+evaluation(cfg,sel,True):
            p=validate_payload(out/'cells'/(unit_key(c)+'.json'),c,frozen,environment,sh if c['phase']=='evaluation' else None);assert p
            sig[unit_key(c)]={k:p[k] for k in ['status','metrics','trace','initial_parameters_sha256','minibatches_sha256','parameter_sha256','completed_steps']}
            if c['phase']=='evaluation':assert p['status']=='COMPLETE'
        signatures.append(sig)
    assert signatures[0]==signatures[1];checks['deterministic_repeat']=True
    terms=[load(p) for p in (ROOT/'outputs/smoke').glob('terminal_*.json') if p.name!='terminal_status.json']
    stops=[t for t in terms if t['status']=='CONTROLLED_STOP' and t['exit_code']==75 and t['new_units']==2]
    assert len(stops)==1
    resumed=load(ROOT/'outputs/smoke/terminal_status.json');assert resumed['resumed_units']==2 and resumed['new_units']==66
    stopped=load(ROOT/'outputs/smoke/STOPPED_TWO_HASHES.json');before=stopped['files']
    assert len(before)==2 and all(sha(ROOT/'outputs/smoke/cells'/name)==h for name,h in before.items())
    assert stopped['attempt_id']==stops[0]['attempt_id'] and stopped['descendants_alive_after_shutdown']==[]
    assert set(stops[0]['completed_unit_ids'])==set(before) and resumed['resumed_files']==before
    assert stops[0]['attempt_id']!=resumed['attempt_id'] and set(before)=={unit_key(c)+'.json' for c in development(cfg,True)[:2]}
    checks['two_checkpoint_hashes_preserved']=True
    events=[]
    for p in (ROOT/'outputs/smoke').glob('heartbeat_*.jsonl'):
        ev=[json.loads(s) for s in p.read_text().splitlines()];events.extend(ev)
        stamps=[datetime.fromisoformat(e['timestamp']).timestamp() for e in ev]
        assert all(0<=b-a<45 for a,b in zip(stamps,stamps[1:]))
    assert all(HB_FIELDS<=e.keys() for e in events)
    assert len({e['completed_units'] for e in events})>=3
    pairs=[]
    for p in sorted((ROOT/'outputs/smoke/worker_heartbeats').glob('*.jsonl')):
        ev=[json.loads(s) for s in p.read_text().splitlines()]
        assert all(HB_FIELDS<=e.keys() for e in ev)
        stamps=[datetime.fromisoformat(e['timestamp']).timestamp() for e in ev]
        assert all(0<=b-a<45 for a,b in zip(stamps,stamps[1:]))
        pair=advancing_pair(ev)
        if pair:pairs.append(pair)
    assert pairs,'NO_SAME_UNIT_TRAINING_ADVANCEMENT'
    after=[pair for pair in pairs if pair[0]['timestamp']>stopped['timestamp']]
    assert after,'NO_POST_RESUME_WORKER_ADVANCEMENT'
    checks['bounded_liveness_and_same_unit_advancement']=True
    out=ROOT/'outputs/calibration';term=load(out/'terminal_status.json')
    assert term['status']=='COMPLETED' and term['exit_code']==0 and term['completed_units']==90
    estimated=0.;timings=[]
    for c in calibration_plan(cfg):
        p=validate_payload(out/'cells'/(unit_key(c)+'.json'),c,frozen,environment);assert p and p['status']=='COMPLETE'
        repeats=12 if c['arm']=='none' else 60
        # 30/3 extrapolation also multiplies I/O/evaluation overhead. A further
        # 2x safety factor cancels ideal two-worker parallelism: conservative.
        estimate=p['runtime_seconds']*10*repeats;estimated+=estimate
        timings.append(dict(dataset=c['dataset'],arm=c['arm'],runtime_seconds=p['runtime_seconds'],estimated_serial_seconds=estimate))
    checks['eta_under_eight_hours']=estimated<cfg['watchdog_seconds']
    evidence_files={p.relative_to(ROOT).as_posix():sha(p) for phase in ['smoke','repeat','calibration']
        for p in (ROOT/'outputs'/phase).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl']}
    body=dict(status='PASS' if all(checks.values()) else 'HOLD',generated_utc=now(),freeze_sha256=frozen,
        environment=environment,checks=checks,conservative_eta_seconds=estimated,calibration_timings=timings,
        evidence_files=evidence_files,same_unit_advancement_examples=[pairs[0],after[0]],
        calibration_role='TIME_ONLY_NON_EVIDENTIARY; no performance-dependent grid or protocol choice',
        review_and_prospective_preflight_still_required=True)
    exclusive(ROOT/'SMOKE_GATE.json',body);print(json.dumps(body))

if __name__=='__main__':main()
