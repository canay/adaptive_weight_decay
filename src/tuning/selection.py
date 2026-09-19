"""Immutable complete-development selection, no test-dependent decisions."""
from core import *
from engine import validate_payload

def select_candidate(payloads):
    finite_candidates=[p for p in payloads if p['status']=='COMPLETE']
    if not finite_candidates:raise RuntimeError('NO_FINITE_CANDIDATE_FOR_CONDITION')
    return min(finite_candidates,key=lambda p:(-p['metrics']['accuracy'],p['cell']['hp']['index']))

def lock_selection(out,cfg,frozen,environment,smoke=False):
    grouped={};hashes={}
    for c in development(cfg,smoke):
        path=out/'cells'/(unit_key(c)+'.json');p=validate_payload(path,c,frozen,environment)
        if p is None:raise RuntimeError('DEVELOPMENT_INCOMPLETE')
        if p['data']['test_file_opened'] or p['data']['evaluation_source']!='validation_from_original_training':
            raise RuntimeError('TEST_LEAKAGE_IN_DEVELOPMENT')
        k=f"{c['dataset']}__n{round(c['noise']*100)}__{c['arm']}"
        grouped.setdefault(k,[]).append(p);hashes[path.name]=sha(path)
    selected={}
    for k,pp in grouped.items():
        chosen=select_candidate(pp)
        selected[k]=dict(hp=chosen['cell']['hp'],validation_accuracy=chosen['metrics']['accuracy'],
            selected_development_payload_sha256=chosen['payload_sha256'],trial_count=len(pp),
            finite_trials=sum(p['status']=='COMPLETE' for p in pp),
            development_cost_seconds=sum(p['runtime_seconds'] for p in pp))
    body=dict(status='LOCKED',freeze_sha256=frozen,environment=environment,smoke=smoke,
        selected=selected,development_files=hashes,metric='validation_accuracy',tie_break='lowest_candidate_index',
        test_access_for_selection=False)
    path=out/'SELECTION.json'
    if path.exists():
        if load(path)!=body:raise RuntimeError('IMMUTABLE_SELECTION_DRIFT')
    else:exclusive(path,body)
    return sha(path),body

def evaluation(cfg,selection,smoke=False):
    ds=['digits'] if smoke else cfg['datasets'];nr=[.2] if smoke else cfg['noise']
    seeds=[931,932] if smoke else cfg['evaluation_seeds'];out=[]
    for d,n,a,s in itertools.product(ds,nr,cfg['arms'],seeds):
        k=f'{d}__n{round(n*100)}__{a}'
        out.append(dict(dataset=d,noise=n,arm=a,hp=selection['selected'][k]['hp'],seed=s,
                        phase='evaluation',smoke=smoke))
    return out
