"""Complete-only, dataset-unit analysis. E0 is explicitly exploratory."""
import argparse, csv, io, itertools, math
import numpy as np
from runner import ROOT, atomic, cells, digest, envelope, key, load, now, valid, verify_bundle

def contrast(values, cfg):
    x=np.asarray(values,dtype=float)
    if len(x)!=9 or not np.isfinite(x).all():raise ValueError('NINE_DATASETS_REQUIRED')
    permutations=np.array(list(itertools.product([-1.,1.],repeat=len(x))))
    estimate=float(x.mean())
    p=float(np.mean(np.abs((permutations*x).mean(1))>=abs(estimate)-1e-14))
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    boot=x[rng.integers(0,len(x),size=(cfg['bootstrap_draws'],len(x)))].mean(1)
    return dict(dataset_effects=x.tolist(),mean=estimate,p_raw=p,
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),n_datasets=len(x),
                test='two_sided_exact_sign_flip_symmetry_assumption',
                bootstrap_seed=cfg['bootstrap_seed'],bootstrap_draws=cfg['bootstrap_draws'])

def holm(rows,cfg):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_raw']); bound=0.
    for rank,i in enumerate(order):
        bound=max(bound,min(1.,(len(rows)-rank)*rows[i]['p_raw']))
        rows[i]['p_holm']=bound
        rows[i]['positive_effect_gate']=bool(rows[i]['mean']>=cfg['effect_floor'] and bound<cfg['alpha'])
        rows[i]['effect_floor']=cfg['effect_floor'];rows[i]['alpha']=cfg['alpha']
    return rows

def e0(cfg):
    source=ROOT/'inputs/previous_raw.csv'
    if digest(source)!='535DF250D5513A5D16C8816C9FAA1AD53A0B293C0055266EB935C5BA9E41D953':
        raise RuntimeError('PREDECESSOR_DRIFT')
    rows=list(csv.DictReader(source.open()))
    if len(rows)!=810:raise RuntimeError('PREDECESSOR_COUNT')
    index={(r['dataset'],r['arm'],int(r['seed'])):float(r['test_acc']) for r in rows}
    if len(index)!=810:raise RuntimeError('PREDECESSOR_DUPLICATE')
    findings=[]
    for arm,baseline in itertools.product(['ctrlA','ctrlB'],['adadecay','awd','swd','cwd']):
        effects=[]
        for d in cfg['datasets']:
            effects.append(np.mean([(index[(d+'_n20',arm,s)]-index[(d+'_n20',baseline,s)])-
                (index[(d+'_full',arm,s)]-index[(d+'_full',baseline,s)]) for s in range(5)]))
        findings.append(dict(arm=arm,baseline=baseline,**contrast(effects,cfg)))
    atomic(ROOT/'analysis/E0_interaction.json',dict(status='COMPLETE',generated_utc=now(),
        role='EXPLORATORY_EXISTING_DATA',source_sha256=digest(source),
        confounding_resolved=False,old_superiority_gate_unchanged='0/12',
        dataset_order=cfg['datasets'],comparisons=holm(findings,cfg)))

def main_analysis(cfg):
    out=ROOT/'outputs/main'; term=load(out/'terminal_status.json')
    frozen=verify_bundle()
    if term['status']!='COMPLETED' or term['exit_code']!=0:raise RuntimeError('NOT_COMPLETE')
    base,controls=cells(cfg);env=term['environment'];index={};raw=[];hashes={};pairing={};noise_pairing={}
    for cell in base+controls:
        path=out/'cells'/(key(cell)+'.json')
        p=valid(path,cell,frozen,env)
        if p is None:raise RuntimeError('MISSING')
        paired_id=(cell['dataset'],cell['size'],cell['noise'],cell['seed'])
        identity={k:p['data'][k] for k in ('selected_indices_sha256','noisy_labels_sha256','features_sha256','clean_labels_sha256')}
        identity.update(init=p['initial_parameters_sha256'],batches=p['minibatches_sha256'])
        if paired_id in pairing and pairing[paired_id]!=identity:raise RuntimeError('ARM_PAIRING_DRIFT')
        pairing[paired_id]=identity
        noise_id=(cell['dataset'],cell['size'],cell['seed'])
        clean_identity={k:v for k,v in identity.items() if k!='noisy_labels_sha256'}
        if noise_id in noise_pairing and noise_pairing[noise_id]!=clean_identity:raise RuntimeError('NOISE_PAIRING_DRIFT')
        noise_pairing[noise_id]=clean_identity
        if cell['control']!='online':
            donor=dict(cell,seed=cell['donor'],control='online',donor=None)
            if p['donor_payload_sha256']!=digest(out/'cells'/(key(donor)+'.json')):
                raise RuntimeError('DONOR_DRIFT')
        index[(cell['dataset'],cell['size'],cell['noise'],cell['arm'],cell['control'],cell['seed'])]=p['row']
        hashes[path.name]=digest(path)
        raw.append(dict(**cell,**p['row'],negative_log_total_shrinkage=p['negative_log_total_shrinkage']))
    def score(d,size,noise,arm,control,s):return index[(d,size,noise,arm,control,s)]['test_acc']
    def interaction(arm,baseline,sizes):
        return [np.mean([score(d,z,.2,arm,'online',s)-score(d,z,.2,baseline,'online',s)
                        -score(d,z,0.,arm,'online',s)+score(d,z,0.,baseline,'online',s)
                        for z,s in itertools.product(sizes,cfg['seeds'])]) for d in cfg['datasets']]
    primary=[];strata=[];threeway=[]
    for arm,b in itertools.product(['ctrlA','ctrlB'],['adadecay','awd','swd','cwd']):
        primary.append(dict(arm=arm,baseline=b,**contrast(interaction(arm,b,cfg['sizes']),cfg)))
        full=interaction(arm,b,['full']);half=interaction(arm,b,['half'])
        for size,effects in [('full',full),('half',half)]:
            strata.append(dict(arm=arm,baseline=b,size=size,**contrast(effects,cfg)))
        threeway.append(dict(arm=arm,baseline=b,**contrast(np.array(half)-full,cfg)))
    feedback=[];timing=[]
    for arm in ['ctrlA','ctrlB']:
        for other in ['online','constant','reverse']:
            effects=[np.mean([score(d,z,n,arm,other,s)-score(d,z,n,arm,'replay',s)
                     for z,n,s in itertools.product(cfg['sizes'],cfg['noise'],cfg['seeds'])]) for d in cfg['datasets']]
            item=dict(arm=arm,contrast=other+'-replay',**contrast(effects,cfg))
            (feedback if other=='online' else timing).append(item)
    # All rows and secondary metrics retained; no selection of favorable cells.
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    target=ROOT/'analysis/main_raw.csv';target.parent.mkdir(exist_ok=True)
    tmp=target.with_suffix('.tmp')
    import os
    with tmp.open('w',encoding='utf-8',newline='') as f:
        f.write(buf.getvalue());f.flush();os.fsync(f.fileno())
    tmp.replace(target)
    atomic(ROOT/'analysis/decision.json',dict(status='COMPLETE',generated_utc=now(),
        role='PROSPECTIVELY_FROZEN_FOLLOWUP_ON_PREVIOUSLY_SEEN_BENCHMARKS',
        freeze_sha256=frozen,environment=env,cell_count=len(raw),dataset_order=cfg['datasets'],
        E1_primary=holm(primary,cfg),E2_primary=holm(feedback,cfg),
        secondary_timing=holm(timing,cfg),exploratory_size_strata=holm(strata,cfg),
        exploratory_threeway=holm(threeway,cfg),source_cells=hashes,
        old_superiority_gate_unchanged='0/12',manuscript_promotion='NOT_AUTHORIZED_BY_THIS_ARTIFACT'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['E0','main'],required=True)
    a=ap.parse_args();cfg=load(ROOT/'config.json');verify_bundle()
    (e0 if a.mode=='E0' else main_analysis)(cfg)
    print('ANALYSIS_COMPLETE '+a.mode)
