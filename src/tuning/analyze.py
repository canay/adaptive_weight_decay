"""Complete-only MC009 E4: tuning-policy noise interactions, twelve-test family."""
import csv,io
from core import *
from engine import validate_payload
from selection import lock_selection,evaluation

def contrast(x,cfg):
    x=np.asarray(x,dtype=float)
    assert x.shape==(9,) and np.isfinite(x).all()
    signs=np.asarray(list(itertools.product([-1,1],repeat=9)))
    boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,9,(cfg['bootstrap_draws'],9))
    return dict(dataset_effects=x.tolist(),mean=float(x.mean()),
        bootstrap95=np.quantile(x[boot].mean(1),[.025,.975]).tolist(),
        p_raw=float(np.mean(np.abs((signs*x).mean(1))>=abs(x.mean())-1e-14)),n_datasets=9)

def holm(rows,cfg):
    last=0.
    for rank,i in enumerate(sorted(range(len(rows)),key=lambda i:rows[i]['p_raw'])):
        last=max(last,min(1.,(len(rows)-rank)*rows[i]['p_raw']))
        rows[i].update(p_holm=last,positive_effect_gate=bool(rows[i]['mean']>=cfg['effect_floor'] and last<cfg['alpha']))
    return rows

def family_summary(rows,cfg):
    expected=2*len(cfg['primary_baselines'])
    if len(rows)!=expected or expected!=12:raise RuntimeError('PRIMARY_FAMILY_INCOMPLETE')
    passed=sum(r['positive_effect_gate'] for r in rows)
    return dict(verdict='FULL_SUPPORT' if passed==expected else 'MIXED_SUPPORT' if passed else 'NO_SUPPORTED_POSITIVE_INTERACTION',
        supported=passed,planned=expected,all_required=passed==expected,
        effect_floor=cfg['effect_floor'],alpha=cfg['alpha'],bootstrap_seed=cfg['bootstrap_seed'],
        bootstrap_draws=cfg['bootstrap_draws'],test='two-sided exact sign-flip; dataset-level symmetry assumption',
        sign_patterns=512,minimum_raw_p=2/512,minimum_first_holm_p=2*expected/512,
        non_significance_is_not_equivalence=True,old_primary_gates_unchanged=True)

def boundary_flags(arm,hp,cfg):
    gg=grid(arm,cfg['lrs']);flags={}
    for key in hp:
        if key=='index':continue
        values=sorted({g[key] for g in gg})
        flags[key]=dict(value=hp[key],minimum=values[0],maximum=values[-1],
            at_minimum=hp[key]==values[0],at_maximum=hp[key]==values[-1],search_axis=len(values)>1)
    return flags

def calculate(index,cfg):
    primary=[]
    for a,b in itertools.product(['ctrlA','ctrlB'],cfg['primary_baselines']):
        gaps=[]
        for n in cfg['noise']:
            gaps.append([float(np.mean([index[d,n,a,s]['accuracy']-index[d,n,b,s]['accuracy'] for s in cfg['evaluation_seeds']])) for d in cfg['datasets']])
        primary.append(dict(arm=a,baseline=b,**contrast(np.asarray(gaps[1])-gaps[0],cfg),
            clean_gap_mean=float(np.mean(gaps[0])),noisy_gap_mean=float(np.mean(gaps[1]))))
    descriptive=[]
    for n,a in itertools.product(cfg['noise'],cfg['arms']):
        pp=[index[d,n,a,s] for d,s in itertools.product(cfg['datasets'],cfg['evaluation_seeds'])]
        descriptive.append(dict(noise=n,arm=a,n_runs=len(pp),**{k:float(np.mean([p[k] for p in pp])) for k in pp[0]}))
    holm(primary,cfg)
    return dict(E4_primary=primary,family_summary=family_summary(primary,cfg),descriptive=descriptive)

def main():
    cfg=load(ROOT/'config.json');frozen=verify_freeze();out=ROOT/'outputs/main'
    term=load(out/'terminal_status.json')
    assert term['status']=='COMPLETED' and term['exit_code']==0 and term['completed_units']==cfg['planned_units']
    sh,sel=lock_selection(out,cfg,frozen,term['environment']);plan=evaluation(cfg,sel)
    rows=[];index={};hashes={};diverged=[];paired={}
    for c in development(cfg)+plan:
        path=out/'cells'/(unit_key(c)+'.json');p=validate_payload(path,c,frozen,term['environment'],sh if c['phase']=='evaluation' else None)
        assert p;hashes[path.name]=sha(path)
        row=dict(dataset=c['dataset'],noise=c['noise'],phase=c['phase'],arm=c['arm'],seed=c['seed'],
            hp_index=c['hp']['index'],hp_json=canonical(c['hp']).decode(),status=p['status'],runtime_seconds=p['runtime_seconds'],
            completed_steps=p['completed_steps'],**{k:None if p['metrics'] is None else p['metrics'][k] for k in ['accuracy','ece','nll','train_accuracy','train_nll','clean_train_accuracy']})
        row['decay_coefficient_mean']=float(np.mean([t['decay']['coefficient_mean'] for t in p['trace']])) if p['trace'] else None
        row['decay_factor_min']=min(t['decay']['factor_min'] for t in p['trace']) if p['trace'] else None
        row['decay_nonpositive_fraction']=float(np.mean([t['decay']['nonpositive_fraction'] for t in p['trace']])) if p['trace'] else None
        logs=[t['decay']['negative_log_abs_factor_mean'] for t in p['trace']]
        row['undefined_log_factor_windows']=sum(v is None for v in logs)
        row['negative_log_abs_factor_mean']=float(np.mean(logs)) if logs and all(v is not None for v in logs) else None
        rows.append(row)
        if c['phase']=='evaluation':
            if p['status']!='COMPLETE':diverged.append(unit_key(c));continue
            identity={k:p['data'][k] for k in ['train_features_sha256','train_labels_sha256','evaluation_features_sha256','evaluation_labels_sha256']}
            identity.update(init=p['initial_parameters_sha256'],batches=p['minibatches_sha256'])
            pk=(c['dataset'],c['noise'],c['seed']);paired.setdefault(pk,identity);assert paired[pk]==identity
            index[c['dataset'],c['noise'],c['arm'],c['seed']]=p['metrics']
    assert len(rows)==cfg['planned_units'] and set(hashes)=={p.name for p in (out/'cells').glob('*.json')}
    dest=ROOT/'analysis';dest.mkdir(exist_ok=True)
    if (dest/'decision.json').exists():
        existing=load(dest/'decision.json')
        assert existing['freeze_sha256']==frozen and existing['selection_sha256']==sh
        assert existing['source_cells']==hashes and existing['all_trials_sha256']==sha(dest/'all_trials.csv')
        print('ALREADY_COMPLETE_VERIFIED');return
    stats={} if diverged else calculate(index,cfg)
    costs=[];boundaries=[]
    for d,n,a in itertools.product(cfg['datasets'],cfg['noise'],cfg['arms']):
        key=f'{d}__n{round(n*100)}__{a}';chosen=sel['selected'][key]
        rr=[r for r in rows if (r['dataset'],r['noise'],r['arm'])==(d,n,a)]
        refit=sum(r['runtime_seconds'] for r in rr if r['phase']=='evaluation')
        costs.append(dict(dataset=d,noise=n,arm=a,all_search_seconds=chosen['development_cost_seconds'],
            all_refit_seconds=refit,refit_count=len(cfg['evaluation_seeds']),mean_refit_seconds=refit/len(cfg['evaluation_seeds']),
            search_plus_one_mean_refit_seconds=chosen['development_cost_seconds']+refit/len(cfg['evaluation_seeds']),
            search_amortized_per_evaluated_refit_seconds=chosen['development_cost_seconds']/len(cfg['evaluation_seeds']),
            trial_count=chosen['trial_count'],finite_trials=chosen['finite_trials']))
        boundaries.append(dict(dataset=d,noise=n,arm=a,axes=boundary_flags(a,chosen['hp'],cfg)))
    frequencies=[]
    for a in cfg['arms']:
        bb=[b for b in boundaries if b['arm']==a]
        for axis in bb[0]['axes']:
            frequencies.append(dict(arm=a,axis=axis,conditions=len(bb),
                at_minimum=sum(b['axes'][axis]['at_minimum'] for b in bb),
                at_maximum=sum(b['axes'][axis]['at_maximum'] for b in bb),search_axis=bb[0]['axes'][axis]['search_axis']))
    body=dict(status='PRIMARY_COMPLETENESS_FAIL' if diverged else 'COMPLETE',generated_utc=now(),
        freeze_sha256=frozen,selection_sha256=sh,run_id=cfg['run_id'],dataset_order=cfg['datasets'],source_cells=hashes,
        development_units=cfg['planned_development'],evaluation_units=cfg['planned_evaluation'],
        development_diverged=sum(r['status']=='DIVERGED' and r['phase']=='development' for r in rows),
        evaluation_diverged=diverged,selected=sel['selected'],**stats,
        search_and_refit_costs=costs,selected_grid_boundaries=boundaries,boundary_frequencies=frequencies,
        boundary_scope='Finite-grid endpoints, not evidence of an optimal boundary or permission to expand the grid',
        role='POST_PRIOR_RESULTS_PROSPECTIVE_TUNING_POLICY_INTERACTION',old_primary_gates_unchanged=True,
        manuscript_promotion='NOT_PERFORMED',runtime_scope='Instrumented CPU implementation, includes diagnostics and evaluation')
    buffer=io.StringIO(newline='');w=csv.DictWriter(buffer,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp=dest/('all_trials.csv.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:f.write(buffer.getvalue().encode('utf-8'));f.flush();os.fsync(f.fileno())
    os.replace(tmp,dest/'all_trials.csv')
    body['all_trials_sha256']=sha(dest/'all_trials.csv')
    atomic(dest/'decision.json',body);print(body['status'])

if __name__=='__main__':main()
