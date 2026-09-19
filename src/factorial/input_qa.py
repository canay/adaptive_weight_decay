"""Pre-training all-context scale check; no label outcomes or model selection."""
from runner import *

def inspect_inputs():
    cfg=load(ROOT/'config.json');rows=[]
    for ds,size,seed in itertools.product(cfg['datasets'],cfg['sizes'],cfg['seeds']):
        clean,_,m=data_for(dict(dataset=ds,size=size,seed=seed,noise=0.))
        noisy,_,n=data_for(dict(dataset=ds,size=size,seed=seed,noise=.2))
        for field in ('features_sha256','test_features_sha256','train_mean_sha256','train_scale_sha256'):
            assert m[field]==n[field], (ds,size,seed,field)
        assert max(m['train_max_abs'],m['test_max_abs'])<1e6, (ds,size,seed,m)
        rows.append(dict(dataset=ds,size=size,seed=seed,**m))
    return dict(status='PASS',created_utc=now(),contexts=len(rows),
                freeze_sha256=verify_bundle(),environment=envelope(),rows=rows)

if __name__=='__main__':
    report=inspect_inputs();atomic(ROOT/'logs/input_qa.json',report)
    print(json.dumps(dict(status=report['status'],contexts=report['contexts'],
        max_train=max(r['train_max_abs'] for r in report['rows']),
        max_test=max(r['test_max_abs'] for r in report['rows']))))
