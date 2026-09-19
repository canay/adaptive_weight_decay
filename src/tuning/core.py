"""Shared immutable-grid and atomic-provenance utilities for MC009.
Date/time: 2026-09-04 16:01 +03:00; Tool: Codex; Model: GPT-5
Operation ID: F06-MC007-LITERATURE-MTA-20260904-01
"""
import hashlib,itertools,json,math,os,platform,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/vendor'))
import common6 as C
from strong_baseline_fidelity import AdamDecayVariant
from pytorch_cpr import AdamCPR as NativeAdamCPR

class AdamCPR(NativeAdamCPR):
    # PyTorch 2.12 renamed this private health-check API. Preserve the official
    # CPR source and all arithmetic; delegate to the installed equivalent guard.
    def _cuda_graph_capture_health_check(self):
        base=torch.optim.Optimizer
        if hasattr(base,'_cuda_graph_capture_health_check'):
            return base._cuda_graph_capture_health_check(self)
        return base._accelerator_graph_capture_health_check(self)
def now():return datetime.now(timezone.utc).isoformat()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def objhash(v):return hashlib.sha256(canonical(v)).hexdigest().upper()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with t.open('xb') as f:f.write(canonical(v));f.flush();os.fsync(f.fileno())
    os.replace(t,p)
    if os.name=='posix':
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
def exclusive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(canonical(v));f.flush();os.fsync(f.fileno())
def env():return dict(host=platform.node(),os_family=platform.system(),architecture=platform.machine(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,device='cpu')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest().upper()
def modelhash(model):return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest().upper()
def grid(arm,lrs):
    wd=[0.,.0001,.001,.01,.1,.3,1.,3.,10.]
    if arm=='none':settings=[dict(scale=0.)]
    elif arm in ['fixed1','cwd']:settings=[dict(scale=v) for v in wd]
    elif arm in ['ctrlA','ctrlB']:settings=[dict(scale=v) for v in [0.,.01,.03,.1,1/3,.5,1.,2.,3.]]
    elif arm=='adadecay':settings=[dict(scale=0.,alpha=4.)]+[dict(scale=v,alpha=a) for v,a in itertools.product([.01,.1,1.,10.],[1.,4.])]
    elif arm=='awd':settings=[dict(scale=v) for v in [0.,.001,.01,.022,.1,1.,10.,30.,100.]]
    elif arm=='swd':settings=[dict(scale=v) for v in [0.,1e-6,1e-5,.0001,.0005,.001,.003,.01,.03]]
    elif arm in ['cpr_native','cpr_all']:settings=[dict(warm_fraction=v) for v in [1.,3/4,1/2,1/4,1/8,1/16,1/32,1/64,1/128]]
    else:raise ValueError(arm)
    return [dict(lr=lr,**h,index=i) for i,(lr,h) in enumerate(itertools.product(lrs,settings))]
def unit_key(c):return f"{c['dataset']}__n{round(c['noise']*100)}__{c['phase']}__{c['arm']}__i{c['hp']['index']}__s{c['seed']}"
def development(cfg,smoke=False):
    ds=['digits'] if smoke else cfg['datasets'];nr=[.2] if smoke else cfg['noise'];out=[]
    for d,n,a in itertools.product(ds,nr,cfg['arms']):
        gg=grid(a,cfg['lrs'])
        if smoke:gg=[g for g in gg if g['index'] in ([0,1,2] if a=='none' else [0,8,13,18,26])]
        for hp in gg:out.append(dict(dataset=d,noise=n,arm=a,hp=hp,seed=930 if smoke else cfg['development_seed'],phase='development',smoke=smoke))
    return out
def verify_freeze():
    f=load(ROOT/'freeze.json')
    for path,digest in f['files'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('SOURCE_INPUT_DRIFT '+path)
    return sha(ROOT/'freeze.json')
def controller_next(lam,signal):return min(30.,max(0.,lam+min(1.,max(-1.,signal))))

def calibration_plan(cfg):
    return [dict(dataset=d,noise=.2,arm=a,hp=grid(a,cfg['lrs'])[1 if a=='none' else 13],
        seed=939,phase='development',smoke=False,calibration=True) for d,a in itertools.product(cfg['datasets'],cfg['arms'])]
