"""MC009 cell engine: development cannot read test; all failures are retained.
Date/time: 2026-09-04 16:12 +03:00; Tool: Codex; Model: GPT-5
Operation ID: F06-MC007-LITERATURE-MTA-20260904-01
"""
import time,threading
from core import *
from data import load_data
from liveness import CellHeartbeat

class NumericalDivergence(Exception):pass

def finite(values,stage):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise NumericalDivergence(stage)

def make_optimizer(model,arm,hp,total_steps):
    lr=hp['lr']
    if arm in ['awd','swd','cwd']:
        strength=hp['scale']/({'awd':.022,'swd':.0005,'cwd':1.}[arm])
        return AdamDecayVariant(model.parameters(),arm,lr,strength=strength)
    if arm.startswith('cpr_'):
        params=model if arm=='cpr_native' else [dict(params=list(model.parameters()),regularize=True)]
        return AdamCPR(params,lr=lr,betas=(.9,.999),eps=1e-8,foreach=False,
            kappa_init_method='warm_start',kappa_init_param=math.ceil(hp['warm_fraction']*total_steps),
            kappa_update=1.,reg_function='l2',reg_by_lr=False)
    return torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,
        weight_decay=0.,foreach=False)

def weighted_diag(parts,lr):
    # parts=(coefficient, coordinate count). CPR uses coefficient=2*lagmul/lr;
    # the native correction is 1-2*lagmul AFTER Adam, not AdamW's pre-update decay.
    parts=[p for p in parts if p[1]>0]
    vals=np.asarray([p[0] for p in parts],dtype=float)
    weights=np.asarray([p[1] for p in parts],dtype=float);weights/=weights.sum()
    factors=1-lr*vals
    if not np.isfinite(vals).all():raise NumericalDivergence('NONFINITE_DECAY_DIAGNOSTIC')
    return dict(coefficient_mean=float(weights@vals),factor_min=float(factors.min()),
        negative_log_abs_factor_mean=None if np.any(factors==0) else float(weights@(-np.log(np.abs(factors)))),
        nonpositive_fraction=float(weights@(factors<=0)))

def cpr_diag(opt,lr):
    parts=[]
    for group in opt.param_groups:
        for p in group['params']:
            st=opt.state[p]
            coeff=2*float(st['lagmul'])/lr if group['regularize'] and float(st['step'])>opt.warm_start else 0.
            parts.append((coeff,p.numel()))
    return weighted_diag(parts,lr)

def optimizer_step(model,opt,arm,hp,lam):
    lr=hp['lr'];scale=hp.get('scale',0.)
    if arm=='adadecay':
        with torch.no_grad():
            theta=[]
            for p in model.parameters():
                g=p.grad.abs();z=(g-g.mean())/(g.std(unbiased=False)+1e-12)
                theta.append(2*torch.sigmoid(hp['alpha']*z))
        opt.step()
        with torch.no_grad():
            for p,th in zip(model.parameters(),theta):p.mul_(1-lr*scale*th)
        coeff=np.concatenate([t.numpy().ravel().astype(np.float64)*scale for t in theta])
        factors=1-lr*coeff
        d=dict(coefficient_mean=float(coeff.mean()),factor_min=float(factors.min()),
            negative_log_abs_factor_mean=None if np.any(factors==0) else float(-np.log(np.abs(factors)).mean()),
            nonpositive_fraction=float(np.mean(factors<=0)))
    elif arm in ['awd','swd','cwd']:
        d0=opt.step();q=d0['cwd_mask_fraction'] if arm=='cwd' else 1.
        d=weighted_diag([(d0['effective_decay'],q),(0.,1-q)],lr)
    elif arm.startswith('cpr_'):
        opt.step();d=cpr_diag(opt,lr)
    else:
        opt.param_groups[0]['weight_decay']=scale*lam
        opt.step();d=weighted_diag([(scale*lam,1)],lr)
    finite(list(model.parameters()),'NONFINITE_PARAMETERS')
    if not all(math.isfinite(v) for v in d.values() if v is not None):
        raise NumericalDivergence('NONFINITE_DIAGNOSTIC')
    return d

def validate_payload(path,cell,frozen,environment,selection_sha=None):
    if not Path(path).exists():return None
    p=load(path);body={k:v for k,v in p.items() if k!='payload_sha256'}
    if (p.get('payload_sha256')!=objhash(body) or p.get('cell')!=cell or
        p.get('freeze_sha256')!=frozen or p.get('environment')!=environment or
        p.get('status') not in ['COMPLETE','DIVERGED'] or p.get('selection_sha256')!=selection_sha):
        raise RuntimeError('INVALID_EXISTING_CELL '+str(path))
    if p['status']=='COMPLETE':
        expected=3 if cell['smoke'] or cell.get('calibration') else load(ROOT/'config.json')['windows']
        if len(p['trace'])!=expected or not all(math.isfinite(v) for v in p['metrics'].values()):
            raise RuntimeError('INVALID_COMPLETE_TRACE')
    elif not p.get('divergence'):raise RuntimeError('DIVERGENCE_WITHOUT_REASON')
    return p

def run_cell(cell,out_s,frozen,cfg,environment,selection_sha=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out_s);target=out/'cells'/(unit_key(cell)+'.json')
    if validate_payload(target,cell,frozen,environment,selection_sha):return unit_key(cell),True
    t0=time.monotonic();hb=CellHeartbeat(out,cell,cfg);live=hb.live;errors=hb.errors
    try:
        tensors,nc,meta=load_data(cell,selection_sha,out/'SELECTION.json')
        x,y,xe,ye,yclean=tensors;model=C.make_mlp(x.shape[1],nc,cell['seed'])
        init=modelhash(model);arm=cell['arm'];hp=cell['hp'];lr=hp['lr']
        steps=2 if cell['smoke'] else math.ceil(meta['n_full']/cfg['batch'])
        windows=3 if cell['smoke'] or cell.get('calibration') else cfg['windows'];total=steps*windows
        live.update(progress_denominator=total);hb.phase('training')
        opt=make_optimizer(model,arm,hp,total)
        gen=torch.Generator().manual_seed(60000+cell['seed']);bh=hashlib.sha256()
        trace=[];lam=1. if arm=='fixed1' else 0.;wn_prev=C.param_norm(model);gbar=0.;prev_loss=None
        status='COMPLETE';divergence=None;metrics=None
        try:
            for window in range(windows):
                internal=lam;losses=[];ds=[]
                for step in range(steps):
                    if time.monotonic()-t0>cfg['cell_timeout_seconds']:raise TimeoutError('CELL_TIMEOUT')
                    if errors:raise RuntimeError('WORKER_HEARTBEAT_FAILED '+str(errors))
                    j=torch.randint(len(x),(cfg['batch'],),generator=gen);bh.update(j.numpy().tobytes())
                    if arm in ['awd','swd','cwd']:opt.zero_grad()
                    else:opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(model(x[j]),y[j]);finite([loss],'NONFINITE_TRAIN_LOSS')
                    loss.backward();finite([p.grad for p in model.parameters()],'NONFINITE_GRADIENT')
                    ds.append(optimizer_step(model,opt,arm,hp,lam));losses.append(float(loss.detach()))
                    live['completed_steps']+=1
                wn=C.param_norm(model)
                if not math.isfinite(wn):raise NumericalDivergence('NONFINITE_WEIGHT_NORM')
                growth=(wn-wn_prev)/(wn_prev+1e-12);wn_prev=wn;gbar=.5*gbar+.5*growth
                tl=float(np.mean(losses));rd=0. if prev_loss is None else (prev_loss-tl)/(prev_loss+1e-8);prev_loss=tl
                if arm in ['ctrlA','ctrlB'] and 2<=window+1<windows:
                    lam=controller_next(lam,gbar/(lr*steps) if arm=='ctrlA' else 10*rd)
                decay={k:(None if any(d[k] is None for d in ds) else float(np.mean([d[k] for d in ds]))) for k in ds[0]}
                decay['factor_min']=min(d['factor_min'] for d in ds)
                trace.append(dict(window=window,steps=steps,lambda_internal=internal,lambda_next=lam,
                    train_loss=tl,weight_norm=wn,norm_signal=gbar/(lr*steps),loss_signal=10*rd,decay=decay))
                hb.emit()
            hb.phase('final_evaluation')
            ev=C.evaluate(model,xe,ye);tr=C.evaluate(model,x,y);cl=C.evaluate(model,x,yclean)
            metrics=dict(accuracy=ev['acc'],ece=ev['ece'],nll=ev['loss'],train_accuracy=tr['acc'],
                train_nll=tr['loss'],clean_train_accuracy=cl['acc'])
            if not all(math.isfinite(v) for v in metrics.values()):raise NumericalDivergence('NONFINITE_FINAL_METRICS')
        except NumericalDivergence as exc:
            status='DIVERGED';metrics=None
            divergence=dict(reason=str(exc),completed_steps=live['completed_steps'],attempted_step=live['completed_steps']+1,
                phase=live['phase'],elapsed_seconds=time.monotonic()-t0)
        if errors:raise RuntimeError('WORKER_HEARTBEAT_FAILED '+str(errors))
        result=dict(status=status,cell=cell,freeze_sha256=frozen,environment=environment,
            attempt_id=live['attempt_id'],pid=os.getpid(),started_utc=live['timestamp'],exit_code=0,
            selection_sha256=selection_sha,completed_utc=now(),data=meta,metrics=metrics,trace=trace,
            divergence=divergence,completed_steps=live['completed_steps'],total_steps=total,
            runtime_seconds=time.monotonic()-t0,initial_parameters_sha256=init,minibatches_sha256=bh.hexdigest().upper(),
            parameter_sha256=modelhash(model),decay_diagnostic_scope='analytic_component_not_total_update; CPR_postAdam_factor_not_AdamW_preAdam')
        result['payload_sha256']=objhash(result);atomic(target,result)
        live.update(completed_atomic_units=1,last_durable_checkpoint_at=now());hb.phase('checkpoint_committed')
        return unit_key(cell),False
    except BaseException as exc:
        exclusive(out/'failed_attempts'/(unit_key(cell)+'__'+live['attempt_id']+'.json'),
            dict(status='FAILED',cell=cell,attempt_id=live['attempt_id'],pid=os.getpid(),freeze_sha256=frozen,
                 environment=environment,completed_utc=now(),error=repr(exc),exit_code=1,
                 completed_steps=live['completed_steps'],scientific_result=False))
        raise
    finally:
        hb.close()
