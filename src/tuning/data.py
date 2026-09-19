"""Partitioned training/validation loader; test requires locked selection.
Date/time: 2026-09-04 16:03 +03:00; Tool: Codex; Model: GPT-5
Operation ID: F06-MC007-LITERATURE-MTA-20260904-01
"""
from core import *
def standardize(x,other):
    mu=x.mean(0);constant=np.ptp(x,axis=0)==0
    scale=np.where(constant,1.,x.std(0)+1e-8)
    xx=((x-mu)/scale).astype(np.float32);oo=((other-mu)/scale).astype(np.float32)
    if not(np.isfinite(xx).all() and np.isfinite(oo).all()):raise RuntimeError('INVALID_SCALED_DATA')
    return xx,oo,dict(mean_sha256=ah(mu),scale_sha256=ah(scale),zero_variance_features=int(constant.sum()),rule='exact_constant_scale1_else_std_plus_1e-8')
def labels(y,nc,noise,seed):
    rng=np.random.RandomState(seed);flip=rng.rand(len(y))<noise
    inc=rng.randint(1,nc,len(y));out=y.copy();out[flip]=(y[flip]+inc[flip])%nc
    return out
def load_data(cell,selection_sha=None,selection_path=None):
    d=cell['dataset'];part=load(ROOT/'inputs'/f'{d}_partition.json')
    with np.load(ROOT/'inputs'/f'{d}_train.npz',allow_pickle=False) as z:
        allx=z['X'].astype(np.float64);ally=z['y'].astype(np.int64);nc=int(z['n_class'])
    yn=labels(ally,nc,cell['noise'],part['noise_seed']);nfull=len(ally)
    if cell['phase']=='development':
        tr=np.asarray(part['development_train_indices'],dtype=np.int64)
        va=np.asarray(part['validation_indices'],dtype=np.int64)
        x=allx[tr];y=yn[tr];other=allx[va];yo=yn[va]
        other_id='validation_from_original_training';ids=va
    elif cell['phase']=='evaluation':
        if not selection_sha or not selection_path or sha(selection_path)!=selection_sha:
            raise RuntimeError('TEST_ACCESS_WITHOUT_LOCKED_SELECTION')
        sel=load(selection_path)
        selection_key=f"{d}__n{round(cell['noise']*100)}__{cell['arm']}"
        if sel['selected'][selection_key]['hp']!=cell['hp']:raise RuntimeError('TEST_CONFIG_NOT_SELECTED')
        tr=np.arange(nfull,dtype=np.int64);x=allx;y=yn
        with np.load(ROOT/'inputs'/f'{d}_test.npz',allow_pickle=False) as z:
            other=z['X'].astype(np.float64);yo=z['y'].astype(np.int64)
        other_id='original_clean_test';ids=np.arange(len(yo),dtype=np.int64)
    else:raise RuntimeError('UNKNOWN_PHASE')
    x,other,scaler=standardize(x,other)
    meta=dict(n_full=nfull,n_train=len(y),n_evaluation=len(yo),n_class=nc,
        input_train_sha256=sha(ROOT/'inputs'/f'{d}_train.npz'),
        partition_sha256=sha(ROOT/'inputs'/f'{d}_partition.json'),
        train_indices_sha256=ah(tr),evaluation_indices_sha256=ah(ids),
        evaluation_source=other_id,full_noisy_labels_sha256=ah(yn),
        train_features_sha256=ah(x),evaluation_features_sha256=ah(other),
        train_labels_sha256=ah(y),evaluation_labels_sha256=ah(yo),
        train_noise_fraction=float(np.mean(y!=ally[tr])),
        selection_sha256=selection_sha,scaler=scaler,
        test_file_opened=cell['phase']=='evaluation')
    return tuple(torch.from_numpy(v) for v in [x,y,other,yo,ally[tr].copy()]),nc,meta
