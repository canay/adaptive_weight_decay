"""Input integrity and train/validation/test roles; no fitted models."""
from core import *
from data import load_data

def main():
    cfg=load(ROOT/'config.json');frozen=verify_freeze();rows=[]
    for d,n in itertools.product(cfg['datasets'],cfg['noise']):
        c=dict(dataset=d,noise=n,phase='development');t,nc,m=load_data(c)
        assert all(torch.isfinite(x).all() for x in t)
        assert not m['test_file_opened'] and m['selection_sha256'] is None
        assert 0<=int(t[1].min())<=int(t[1].max())<nc
        assert 0<=int(t[3].min())<=int(t[3].max())<nc
        rows.append(dict(dataset=d,noise=n,**m))
    result=dict(status='PASS',freeze_sha256=frozen,environment=env(),rows=rows,
        scope='18 development/noise contexts, labels/scaler/partition integrity; test outcome not accessed')
    exclusive(ROOT/'INPUT_QA.json',result);print('INPUT_QA_PASS',len(rows))

if __name__=='__main__':main()
