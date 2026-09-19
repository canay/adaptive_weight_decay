"""F06-MC-007 isolated, hash-bound safe-scaling factorial replication."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, itertools, json, math, os
import platform, shutil, signal, sys, tempfile, threading, time, traceback, uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'vendor'))
import common6 as C
from strong_baseline_fidelity import AdamDecayVariant

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def objhash(obj):
    return hashlib.sha256(canonical(obj)).hexdigest().upper()

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with tmp.open('wb') as f:
        f.write(canonical(obj)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def output_bytes(out):
    # Atomic temporary siblings may vanish between enumeration and stat.
    return sum(p.stat().st_size for p in out.rglob('*')
               if p.suffix!='.tmp' and p.is_file())

def envelope():
    return dict(os_family=platform.system(), architecture=platform.machine(),
                python_version=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, psutil=psutil.__version__, device='cpu')

def admissible_predecessor(terminal):
    return ((terminal.get('status')=='CONTROLLED_STOP' and terminal.get('exit_code')==75)
            or (terminal.get('status')=='COMPLETED' and terminal.get('exit_code')==0))

def verify_bundle():
    manifest = load(ROOT / 'freeze.json')
    for name, sha in manifest['files'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError('SOURCE_OR_INPUT_DRIFT: ' + name)
    return digest(ROOT / 'freeze.json')

def cells(cfg, smoke=False):
    ds = ['digits'] if smoke else cfg['datasets']
    sizes = ['half'] if smoke else cfg['sizes']
    noise = [.2] if smoke else cfg['noise']
    seeds = [900, 901] if smoke else cfg['seeds']
    base, controls = [], []
    for d, size, p, seed in itertools.product(ds, sizes, noise, seeds):
        common = dict(dataset=d, size=size, noise=p, seed=seed,
                      windows=3 if smoke else cfg['windows'], smoke=smoke)
        for arm in cfg['arms']:
            base.append(dict(**common, arm=arm, control='online', donor=None))
        donor = seeds[(seeds.index(seed)+1) % len(seeds)]
        for arm, mode in itertools.product(['ctrlA','ctrlB'], cfg['control_types']):
            controls.append(dict(**common, arm=arm, control=mode, donor=donor))
    return base, controls

def key(cell):
    return (f"{cell['dataset']}__{cell['size']}__n{round(cell['noise']*100)}__"
            f"{cell['arm']}__{cell['control']}__s{cell['seed']}")

def valid(path, cell, root_hash, env):
    if not path.exists():
        return None
    p = load(path)
    body = {k:v for k,v in p.items() if k != 'payload_sha256'}
    if (p.get('payload_sha256') != objhash(body) or p.get('cell') != cell
        or p.get('freeze_sha256') != root_hash or p.get('environment') != env
        or p.get('status') != 'COMPLETE'
        or len(p.get('trace', [])) != cell['windows']):
        raise RuntimeError('INVALID_EXISTING_CELL: ' + str(path))
    for metric in ('test_acc','test_ece','test_loss','train_acc','clean_train_acc'):
        if not math.isfinite(p['row'][metric]):
            raise RuntimeError('NONFINITE_EXISTING_CELL')
    return p

def standardize_train_only(train, test):
    # Exact constancy avoids a roundoff-sized std on repeated float values.
    mu = train.mean(0)
    constant = np.ptp(train, axis=0) == 0
    scale = np.where(constant, 1., train.std(0) + 1e-8)
    x = ((train - mu) / scale).astype(np.float32)
    xt = ((test - mu) / scale).astype(np.float32)
    return x, xt, mu, scale, constant


def data_for(cell):
    with np.load(ROOT/'inputs'/f"{cell['dataset']}_full.npz", allow_pickle=False) as z:
        x = z['Xtr'].astype(np.float64); y = z['ytr'].astype(np.int64)
        xt = z['Xte'].astype(np.float64); yt = z['yte'].astype(np.int64)
        nc = int(z['n_class'])
    nfull = len(y)
    order = np.random.RandomState(110000+cell['seed']).permutation(nfull)
    idx = order if cell['size']=='full' else order[:nfull//2]
    nr = np.random.RandomState(120000+cell['seed'])
    flip = nr.rand(nfull) < cell['noise']
    increments = nr.randint(1, nc, nfull)
    corrupted = y.copy(); corrupted[flip] = (y[flip]+increments[flip]) % nc
    x = x[idx]; yc = y[idx].copy(); yn = corrupted[idx].copy()
    x, xt, mu, scale, constant = standardize_train_only(x, xt)
    if not (np.isfinite(x).all() and np.isfinite(xt).all()):
        raise RuntimeError('DATA_NONFINITE')
    meta = dict(n_full=nfull, n_train=len(idx), n_test=len(yt),
                realized_noise=float(np.mean(yc!=yn)),
                selected_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
                noisy_labels_sha256=hashlib.sha256(yn.tobytes()).hexdigest(),
                features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                clean_labels_sha256=hashlib.sha256(yc.tobytes()).hexdigest(),
                zero_variance_features=int(np.sum(constant)),
                test_features_sha256=hashlib.sha256(xt.tobytes()).hexdigest(),
                train_mean_sha256=hashlib.sha256(mu.tobytes()).hexdigest(),
                train_scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),
                train_max_abs=float(np.max(np.abs(x))),
                test_max_abs=float(np.max(np.abs(xt))),
                scaling_rule='train_exact_constant_scale1_else_std_plus_1e-8')
    return tuple(torch.from_numpy(a) for a in (x,yn,xt,yt,yc)), nc, meta

def run_cell(cell, out_s, freeze_hash, cfg, env):
    torch.set_num_threads(cfg['threads_per_worker'])
    torch.use_deterministic_algorithms(True)
    out = Path(out_s)
    target = out/'cells'/(key(cell)+'.json')
    if valid(target, cell, freeze_hash, env):
        return key(cell), True
    t0=time.monotonic(); proc=psutil.Process()
    tensors, nc, meta = data_for(cell)
    x, y, xt, yt, yc = tensors
    model = C.make_mlp(x.shape[1], nc, cell['seed'])
    init_sha=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    batch_hasher=hashlib.sha256()
    steps=math.ceil(meta['n_full']/cfg['batch'])
    if cell['smoke']:
        steps=2
    arm, mode, lr=cell['arm'], cell['control'], cfg['lr']
    donor_hash=None; schedule=None
    if mode!='online':
        donor_cell=dict(cell, seed=cell['donor'], control='online', donor=None)
        donor_path=out/'cells'/(key(donor_cell)+'.json')
        donor=valid(donor_path, donor_cell, freeze_hash, env)
        if donor is None: raise RuntimeError('DONOR_NOT_COMPLETE')
        donor_hash=digest(donor_path)
        schedule=np.array([r['lambda_applied'] for r in donor['trace']],dtype=float)
        if mode=='reverse': schedule=schedule[::-1].copy()
        if mode=='constant':
            value=-math.expm1(float(np.log1p(-lr*schedule).mean()))/lr
            schedule=np.full(cell['windows'], value)
    strong=arm in ('awd','swd','cwd')
    lam=1.0 if arm=='fixed1' else 0.0
    opt=(AdamDecayVariant(model.parameters(), arm, lr) if strong else
         torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.999),
                           eps=1e-8, weight_decay=lam, foreach=False))
    gen=torch.Generator().manual_seed(60000+cell['seed'])
    trace=[]; wn_prev=C.param_norm(model); gbar=0.; prev_loss=None
    for window in range(cell['windows']):
        if schedule is not None: lam=float(schedule[window])
        if not strong: opt.param_groups[0]['weight_decay']=lam
        applied=lam; losses=[]; diagnostics=[]
        for step in range(steps):
            if time.monotonic()-t0>cfg['cell_timeout_seconds']:
                raise TimeoutError('CELL_TIMEOUT')
            j=torch.randint(len(x),(cfg['batch'],),generator=gen)
            batch_hasher.update(j.numpy().tobytes())
            if strong: opt.zero_grad()
            else: opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[j]),y[j])
            if not torch.isfinite(loss): raise FloatingPointError('TRAIN_NONFINITE')
            loss.backward()
            if arm=='adadecay':
                with torch.no_grad():
                    theta=[]
                    for p in model.parameters():
                        g=p.grad.abs(); z=(g-g.mean())/(g.std(unbiased=False)+1e-12)
                        theta.append(2.0/(1.0+torch.exp(-4.0*z)))
                opt.step()
                with torch.no_grad():
                    for p, th in zip(model.parameters(),theta): p.mul_(1-lr*th)
            elif strong: diagnostics.append(opt.step())
            else: opt.step()
            losses.append(float(loss.detach()))
        wn=C.param_norm(model); growth=(wn-wn_prev)/(wn_prev+1e-12)
        wn_prev=wn; gbar=.5*gbar+.5*growth
        tl=float(np.mean(losses)); rd=0. if prev_loss is None else (prev_loss-tl)/(prev_loss+1e-8)
        prev_loss=tl
        if arm in ('ctrlA','ctrlB') and mode=='online' and 2<=window+1<cell['windows']:
            sig=gbar/(lr*steps) if arm=='ctrlA' else 10*rd
            lam=min(30.,max(0.,lam+min(1.,max(-1.,sig))))
        tr=dict(window=window, steps=steps, lambda_applied=applied,
                lambda_next=lam, train_loss=tl, weight_norm=wn,
                norm_signal=gbar/(lr*steps), loss_signal=10*rd)
        if diagnostics:
            tr['effective_decay_mean']=float(np.mean([v['effective_decay'] for v in diagnostics]))
        trace.append(tr)
        atomic(out/'live'/f'{os.getpid()}.json',dict(
            timestamp=now(),unit_id=key(cell),pid=os.getpid(),phase='training',
            progress_numerator=window+1,progress_denominator=cell['windows'],
            phase_elapsed_seconds=time.monotonic()-t0,
            process_cpu_seconds=sum(proc.cpu_times()[:2]),rss_bytes=proc.memory_info().rss))
    # Final-only evaluation. No metric selects a checkpoint or hyperparameter.
    et=C.evaluate(model,xt,yt); trn=C.evaluate(model,x,y); clean=C.evaluate(model,x,yc)
    row=dict(test_acc=et['acc'],test_ece=et['ece'],test_loss=et['loss'],
             train_acc=trn['acc'],clean_train_acc=clean['acc'],train_loss=trn['loss'],
             runtime_seconds=time.monotonic()-t0)
    if not all(math.isfinite(v) for v in row.values()): raise FloatingPointError('FINAL_NONFINITE')
    parameter_sha=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    scalar = not strong and arm!='adadecay'
    shrinkage = float(sum(-steps*math.log1p(-lr*h['lambda_applied']) for h in trace)) if scalar else None
    result=dict(status='COMPLETE',cell=cell,freeze_sha256=freeze_hash,environment=env,
                completed_utc=now(),pid=os.getpid(),data=meta,row=row,trace=trace,
                donor_payload_sha256=donor_hash,parameter_sha256=parameter_sha,
                initial_parameters_sha256=init_sha,minibatches_sha256=batch_hasher.hexdigest(),
                negative_log_total_shrinkage=shrinkage)
    result['payload_sha256']=objhash(result); atomic(target,result)
    return key(cell),False

def run(args):
    cfg=load(ROOT/'config.json'); frozen=verify_bundle(); env=envelope()
    if args.phase=='main':
        gate=load(ROOT/'SMOKE_GATE.json')
        if gate.get('status')!='PASS' or gate.get('freeze_sha256')!=frozen or gate.get('environment')!=env:
            raise RuntimeError('MAIN_SMOKE_GATE_NOT_CURRENT')
    out=ROOT/'outputs'/args.phase; out.mkdir(parents=True,exist_ok=True)
    terminal=out/'terminal_status.json'
    if terminal.exists() and not admissible_predecessor(load(terminal)):
        raise RuntimeError('PREDECESSOR_FAILED_REQUIRES_EXPLICIT_REPAIR')
    lock=Path(tempfile.gettempdir())/f"f06mc007_{args.phase}.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    attempt=uuid.uuid4().hex; start=time.monotonic(); shutdown=threading.Event()
    base,controls=cells(cfg,args.phase!='main'); plan=base+controls
    state=dict(status='RUNNING',run_id=cfg['run_id'],attempt_id=attempt,pid=os.getpid(),
               phase=args.phase,planned_units=len(plan),completed_units=0,
               freeze_sha256=frozen,environment=env,started_utc=now())
    stop_code=0; executor=None; active={}; submitted={}; resumed=0; new=0; monitor_errors=[]
    sample_lock=threading.RLock()
    def sample():
        with sample_lock:sample_unlocked()
    def sample_unlocked():
        root=psutil.Process(); family=[root]+root.children(recursive=True)
        cpus=[]; rss=0
        for p in family:
            try: cpus.append(dict(pid=p.pid,cpu_seconds=sum(p.cpu_times()[:2]))); rss+=p.memory_info().rss
            except psutil.Error: pass
        event=dict(state,timestamp=now(),elapsed_seconds=time.monotonic()-start,
                   unit_id='supervisor:'+args.phase,unit_elapsed_seconds=time.monotonic()-start,
                   process_tree=cpus,process_tree_rss_bytes=rss,
                   active_units=[key(v) for v in tuple(active.values())],free_bytes=shutil.disk_usage(ROOT).free)
        atomic(out/'heartbeat.json',event)
        with (out/f'heartbeat_{attempt}.jsonl').open('ab') as f: f.write(canonical(event)+b'\n')
    def monitor():
        try:
            while not shutdown.wait(cfg['heartbeat_seconds']): sample()
        except BaseException as exc:
            monitor_errors.append(repr(exc))
    thread=threading.Thread(target=monitor,daemon=True)
    def terminated(signum,frame): raise KeyboardInterrupt(f'SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,terminated)
    try:
        if shutil.disk_usage(ROOT).free<cfg['disk_floor_bytes']: raise RuntimeError('DISK_FLOOR')
        if psutil.virtual_memory().available<4*1024**3: raise RuntimeError('RAM_FLOOR')
        sample(); thread.start()
        for phase_cells in (base,controls):
            todo=[]
            for cell in phase_cells:
                if valid(out/'cells'/(key(cell)+'.json'),cell,frozen,env): resumed+=1
                else: todo.append(cell)
            state['completed_units']=resumed+new
            # Submission is bounded: controlled stop cannot leave unseen extra work.
            if args.stop_after: todo=todo[:max(0,args.stop_after-new)]
            executor=concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'])
            queue=iter(todo)
            while True:
                while len(active)<cfg['workers']:
                    cell=next(queue,None)
                    if cell is None: break
                    fut=executor.submit(run_cell,cell,str(out),frozen,cfg,env); active[fut]=cell
                    submitted[fut]=time.monotonic()
                if not active: break
                done,_=concurrent.futures.wait(active,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    unit,cached=f.result(); del active[f]; del submitted[f]; new+=not cached
                    state['completed_units']=resumed+new
                    atomic(out/'progress.json',dict(state,timestamp=now(),last_completed_unit=unit))
                    sample()
                    print(f"{state['completed_units']}/{len(plan)} {unit}",flush=True)
                if time.monotonic()-start>cfg['watchdog_seconds']: raise TimeoutError('RUN_WATCHDOG')
                if any(time.monotonic()-t>cfg['cell_timeout_seconds']+30 for t in submitted.values()):
                    raise TimeoutError('OPAQUE_CELL_SUPERVISOR_TIMEOUT')
                if monitor_errors: raise RuntimeError('HEARTBEAT_FAILED: '+str(monitor_errors))
                if shutil.disk_usage(ROOT).free<cfg['disk_floor_bytes']: raise RuntimeError('DISK_FLOOR_RUNTIME')
                if psutil.virtual_memory().available<2*1024**3: raise RuntimeError('RAM_FLOOR_RUNTIME')
                if output_bytes(out)>cfg['run_output_budget_bytes']:
                    raise RuntimeError('OUTPUT_DISK_BUDGET')
            executor.shutdown(); executor=None
            if args.stop_after and new>=args.stop_after:
                state['status']='CONTROLLED_STOP'; stop_code=75; break
        else:
            for cell in plan:
                if not valid(out/'cells'/(key(cell)+'.json'),cell,frozen,env): raise RuntimeError('INCOMPLETE')
            state['status']='COMPLETED'
    except BaseException as e:
        stop_code=130 if isinstance(e,KeyboardInterrupt) else 1
        state['status']='TERMINATED' if stop_code==130 else 'FAILED'
        state['error']=repr(e); traceback.print_exc()
    finally:
        if executor is not None:
            # Only this launch's descendants, never other user's processes.
            children=psutil.Process().children(recursive=True)
            for p in children:
                try:p.terminate()
                except psutil.Error:pass
            _,alive=psutil.wait_procs(children,timeout=5)
            for p in alive:
                try:p.kill()
                except psutil.Error:pass
            executor.shutdown(wait=False,cancel_futures=True)
        shutdown.set()
        if thread.is_alive():thread.join(timeout=5)
        state.update(exit_code=stop_code,ended_utc=now(),elapsed_seconds=time.monotonic()-start,
                     resumed_units=resumed,new_units=new)
        atomic(out/'terminal_status.json',state)
        atomic(out/f'terminal_{attempt}.json',state)
        lock.unlink(missing_ok=True)
    return stop_code

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','repeat','main'],required=True)
    ap.add_argument('--stop-after',type=int,default=0)
    sys.exit(run(ap.parse_args()))
