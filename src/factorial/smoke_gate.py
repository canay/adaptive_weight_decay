"""Technical advancement only; never grants scientific or manuscript closure."""
from runner import *

def main():
    cfg=load(ROOT/'config.json');frozen=verify_bundle();env=envelope()
    tests=load(ROOT/'logs/unit_test_receipt.json')
    assert tests['status']=='PASS' and tests['tests_run']==12
    qa=load(ROOT/'logs/input_qa.json')
    assert qa['status']=='PASS' and qa['contexts']==90 and qa['freeze_sha256']==frozen
    assert tests['freeze_sha256']==frozen and tests['environment']==env
    a,b=cells(cfg,True);plan=a+b
    smoke=ROOT/'outputs/smoke';repeat=ROOT/'outputs/repeat'
    terminal=load(smoke/'terminal_status.json');term2=load(repeat/'terminal_status.json')
    assert terminal['status']==term2['status']=='COMPLETED'
    assert terminal['resumed_units']==2 and terminal['new_units']==26
    stop=load(ROOT/'logs/smoke_stop_receipt.json')
    assert stop['terminal']['status']=='CONTROLLED_STOP' and stop['terminal']['exit_code']==75
    assert stop['terminal']['attempt_id']!=terminal['attempt_id']
    for name,sha in stop['files'].items():assert digest(smoke/'cells'/name)==sha
    for cell in plan:
        p=valid(smoke/'cells'/(key(cell)+'.json'),cell,frozen,env)
        q=valid(repeat/'cells'/(key(cell)+'.json'),cell,frozen,env)
        assert p['parameter_sha256']==q['parameter_sha256']
        assert p['trace']==q['trace']
        assert {k:v for k,v in p['row'].items() if k!='runtime_seconds'}=={k:v for k,v in q['row'].items() if k!='runtime_seconds'}
    events=[]
    for file in smoke.glob('heartbeat_*.jsonl'):
        events.extend(json.loads(s) for s in file.read_text().splitlines())
    assert len({e['completed_units'] for e in events})>=3
    assert any(any(p['cpu_seconds']>0 for p in e['process_tree']) for e in events)
    # Full-budget non-evidentiary calibration uses extreme task size and slow arms.
    measurements=[]
    for arm in ['awd','adadecay']:
        cell=dict(dataset='adult',size='full',noise=.2,seed=990,windows=cfg['windows'],
                  smoke=False,arm=arm,control='online',donor=None)
        out=ROOT/'outputs/calibration'
        run_cell(cell,str(out),frozen,cfg,env)
        p=load(out/'cells'/(key(cell)+'.json'));measurements.append(p['row']['runtime_seconds'])
    eta=max(measurements)*2520*1.5  # deliberately serial despite two workers
    assert eta<cfg['watchdog_seconds'], ('ETA_WATCHDOG',eta,cfg['watchdog_seconds'])
    atomic(ROOT/'SMOKE_GATE.json',dict(status='PASS',created_utc=now(),freeze_sha256=frozen,
        environment=env,deterministic_cells=28,resumed_without_change=2,
        heartbeat_advancement_samples=len(events),unit_tests='logs/unit_test_receipt.json:12/12',
        input_qa_sha256=digest(ROOT/'logs/input_qa.json'),
        unit_test_receipt_sha256=digest(ROOT/'logs/unit_test_receipt.json'),
        calibration_seconds=measurements,eta_conservative_seconds=eta,
        partial_result_promotion_policy='prohibited'))
    print('SMOKE_GATE_PASS eta_conservative_seconds='+str(round(eta)))

if __name__=='__main__':main()
