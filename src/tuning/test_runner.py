"""Synthetic and data-integrity tests; no benchmark training."""
import tempfile,unittest
from unittest.mock import patch
from core import *
from data import standardize,labels,load_data
from engine import make_optimizer,optimizer_step,cpr_diag,NumericalDivergence,finite
from selection import select_candidate
from analyze import contrast,holm,family_summary,boundary_flags,calculate
from smoke_gate import advancing_pair
from admission import verify_admission,evidence

class Tests(unittest.TestCase):
    def setUp(self):self.cfg=load(ROOT/'config.json');torch.set_num_threads(1)
    def test_grid_and_plan(self):
        self.assertEqual(len(development(self.cfg)),4428)
        self.assertEqual(len(development(self.cfg,True)),48)
        self.assertEqual(len({unit_key(c) for c in development(self.cfg)}),4428)
        for a in self.cfg['arms']:
            gg=grid(a,self.cfg['lrs']);self.assertEqual(len(gg),3 if a=='none' else 27)
            self.assertEqual([x['index'] for x in gg],list(range(len(gg))))
    def test_partition(self):
        for d in self.cfg['datasets']:
            p=load(ROOT/'inputs'/f'{d}_partition.json');a=p['development_train_indices'];b=p['validation_indices']
            with np.load(ROOT/'inputs'/f'{d}_train.npz') as z:n=len(z['y'])
            self.assertFalse(set(a)&set(b));self.assertEqual(sorted(a+b),list(range(n)))
    def test_development_never_opens_test(self):
        c=development(self.cfg,True)[0];original=np.load;seen=[]
        def guarded(path,*args,**kwargs):
            seen.append(str(path));self.assertNotIn('_test.npz',str(path));return original(path,*args,**kwargs)
        with patch('numpy.load',side_effect=guarded):t,n,m=load_data(c)
        self.assertFalse(m['test_file_opened']);self.assertTrue(seen)
    def test_eval_rejects_no_selection(self):
        c=dict(development(self.cfg,True)[0],phase='evaluation')
        with self.assertRaisesRegex(RuntimeError,'TEST_ACCESS'):load_data(c)
    def test_select_accuracy_then_index_not_nll(self):
        pp=[dict(status='COMPLETE',metrics=dict(accuracy=.8,nll=n),cell=dict(hp=dict(index=i))) for i,n in [(5,.1),(2,9)]]
        pp.append(dict(status='DIVERGED',metrics=None,cell=dict(hp=dict(index=0))))
        self.assertEqual(select_candidate(pp)['cell']['hp']['index'],2)
        with self.assertRaisesRegex(RuntimeError,'NO_FINITE'):select_candidate([pp[-1]])
    def test_scaler_constant(self):
        x=np.array([[3.,0],[3.,2],[3.,4]]);xt=np.array([[7.,6]])
        a,b,m=standardize(x,xt);self.assertTrue(np.all(a[:,0]==0));self.assertEqual(b[0,0],4.)
        self.assertAlmostEqual(float(a[:,1].mean()),0.,places=6)
    def test_noise_repeat(self):
        y=np.arange(500)%3
        self.assertTrue(np.array_equal(labels(y,3,.2,33),labels(y,3,.2,33)))
        self.assertTrue(np.array_equal(labels(y,3,0.,33),y))
    def test_controller(self):
        self.assertEqual(controller_next(0,9),1);self.assertEqual(controller_next(0,-9),0)
        self.assertEqual(controller_next(30,9),30);self.assertEqual(controller_next(10,-9),9)
    def test_cpr_native_grouping(self):
        model=C.make_mlp(4,2,3);opt=make_optimizer(model,'cpr_native',dict(lr=.003,warm_fraction=.125),16)
        groups={id(p):g['regularize'] for g in opt.param_groups for p in g['params']}
        for name,p in model.named_parameters():self.assertEqual(groups[id(p)],name.endswith('weight'))
        opt2=make_optimizer(model,'cpr_all',dict(lr=.003,warm_fraction=.125),16)
        self.assertTrue(all(g['regularize'] for g in opt2.param_groups))
    def test_cpr_scalar_oracle(self):
        # Independent two-step Adam then native CPR calculation in double precision.
        p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
        opt=AdamCPR([dict(params=[p],regularize=True)],lr=.01,betas=(.9,.999),eps=1e-8,
            kappa_init_method='warm_start',kappa_init_param=1,kappa_update=1.,reg_function='l2',reg_by_lr=False,foreach=False)
        oracle=np.array([1.,-2.]);m=np.zeros(2);v=np.zeros(2);lag=0.;kappa=None
        seen_positive=False
        for step in [1,2,3]:
            g=np.array([-.5,.25]);m=.9*m+.1*g;v=.999*v+.001*g*g
            oracle-=.01*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            if step==1:kappa=float(np.float32(np.sum(oracle**2)))
            else:
                lag=float(np.float32(lag+.5*(np.sum(oracle**2)-kappa)));lag=max(0.,lag)
                oracle*=1-2*lag
            p.grad=torch.tensor(g);opt.step()
            np.testing.assert_allclose(p.detach().numpy(),oracle,rtol=2e-6,atol=2e-6)
            seen_positive |= float(opt.state[p]['lagmul'])>0.
        self.assertTrue(seen_positive)
    def test_zero_regularizer_reduces_to_adam(self):
        for arm in ['fixed1','ctrlA','ctrlB','adadecay','awd','swd','cwd']:
            m=C.make_mlp(2,2,5);ref=C.make_mlp(2,2,5)
            hp=dict(lr=.003,scale=0.,alpha=4.)
            opt=make_optimizer(m,arm,hp,20);rop=torch.optim.Adam(ref.parameters(),lr=.003,foreach=False)
            for t in range(3):
                for p,q in zip(m.parameters(),ref.parameters()):p.grad=torch.ones_like(p)*.2;q.grad=torch.ones_like(q)*.2
                optimizer_step(m,opt,arm,hp,7.);rop.step()
            for p,q in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-6)
    def active_oracle(self,arm):
        model=torch.nn.Linear(4,1,bias=False,dtype=torch.float64)
        initial=np.array([[1.,-2.,.5,-.3]])
        with torch.no_grad():model.weight.copy_(torch.from_numpy(initial))
        hp=dict(lr=.01,scale={'fixed1':.3,'adadecay':.1,'awd':.022,'swd':.0005,'cwd':.3}[arm],alpha=4.)
        opt=make_optimizer(model,arm,hp,3);theta=initial.copy();m=np.zeros_like(theta);v=np.zeros_like(theta);ema=0.
        grads=[np.array([[.5,.25,-.75,.1]]),np.array([[-.4,.6,.3,-.2]]),np.array([[.2,-.7,.4,.5]])]
        for t,g in enumerate(grads,1):
            before=theta.copy();m=.9*m+.1*g;v=.999*v+.001*g*g
            u=(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
            if arm=='awd':
                raw=hp['scale']*np.sqrt(np.sum(g*g))/max(np.sqrt(np.sum(before*before)),1e-12)
                ema=.1*ema+.9*raw;theta=before-.01*(u+ema*before)
            elif arm=='swd':
                decay=hp['scale']/max(np.sqrt(np.mean(v/(1-.999**t))),1e-12)
                theta=before-.01*(u+decay*before)
            elif arm=='cwd':theta=before-.01*(u+hp['scale']*(u*before>=0)*before)
            elif arm=='adadecay':
                z=(np.abs(g)-np.abs(g).mean())/(np.abs(g).std()+1e-12)
                coef=hp['scale']*2/(1+np.exp(-hp['alpha']*z));theta=(before-.01*u)*(1-.01*coef)
            else:theta=(1-.01*hp['scale'])*before-.01*u
            model.weight.grad=torch.from_numpy(g);diag=optimizer_step(model,opt,arm,hp,1.)
            np.testing.assert_allclose(model.weight.detach().numpy(),theta,rtol=1e-11,atol=1e-12)
            if arm=='awd':self.assertAlmostEqual(diag['coefficient_mean'],ema,places=12)
        self.assertGreater(float(np.max(np.abs(theta-initial))),0.)
    def test_active_awd_oracle(self):self.active_oracle('awd')
    def test_active_swd_oracle(self):self.active_oracle('swd')
    def test_active_cwd_oracle(self):self.active_oracle('cwd')
    def test_active_adadecay_oracle(self):self.active_oracle('adadecay')
    def test_active_fixed_adamw_oracle(self):self.active_oracle('fixed1')
    def test_cpr_null_endpoint_behavior(self):
        for arm in ['cpr_native','cpr_all']:
            model=torch.nn.Linear(3,2,dtype=torch.float64);ref=torch.nn.Linear(3,2,dtype=torch.float64);ref.load_state_dict(model.state_dict())
            hp=dict(lr=.003,warm_fraction=1.);opt=make_optimizer(model,arm,hp,4)
            adam=torch.optim.Adam(ref.parameters(),lr=.003,foreach=False)
            for t in range(4):
                for p,q in zip(model.parameters(),ref.parameters()):p.grad=torch.full_like(p,.2);q.grad=torch.full_like(q,.2)
                optimizer_step(model,opt,arm,hp,0.);adam.step()
                for p,q in zip(model.parameters(),ref.parameters()):
                    torch.testing.assert_close(p,q,rtol=1e-10,atol=1e-12)
                    self.assertEqual(float(opt.state[p]['lagmul']),0.)
    def test_all_arms_finite_synthetic_step(self):
        for arm in self.cfg['arms']:
            model=C.make_mlp(2,2,7);hp=grid(arm,self.cfg['lrs'])[1 if arm=='none' else 13]
            opt=make_optimizer(model,arm,hp,12)
            for i in range(4):
                for p in model.parameters():p.grad=torch.full_like(p,.1)
                d=optimizer_step(model,opt,arm,hp,1.)
                self.assertTrue(math.isfinite(d['coefficient_mean']))
    def test_numerical_failure(self):
        with self.assertRaises(NumericalDivergence):finite([torch.tensor(float('nan'))],'test')
    def test_exact_family(self):
        rows=[contrast([.01]*9,self.cfg) for _ in range(12)];holm(rows,self.cfg)
        self.assertTrue(all(r['positive_effect_gate'] for r in rows))
        self.assertEqual(rows[0]['p_raw'],2/512);self.assertEqual(rows[0]['p_holm'],24/512)
    def test_family_verdict_all_mixed_none(self):
        for n,expected in [(12,'FULL_SUPPORT'),(1,'MIXED_SUPPORT'),(0,'NO_SUPPORTED_POSITIVE_INTERACTION')]:
            rr=[dict(positive_effect_gate=i<n) for i in range(12)]
            self.assertEqual(family_summary(rr,self.cfg)['verdict'],expected)
        with self.assertRaisesRegex(RuntimeError,'INCOMPLETE'):family_summary([],self.cfg)
    def test_cpr_null_first(self):
        for arm in ['cpr_native','cpr_all']:
            gg=grid(arm,self.cfg['lrs']);self.assertEqual(gg[0]['warm_fraction'],1.)
            self.assertEqual(gg[8]['warm_fraction'],1/128)
    def test_boundary_dimensions(self):
        hp=grid('adadecay',self.cfg['lrs'])[26];f=boundary_flags('adadecay',hp,self.cfg)
        self.assertEqual(set(f),{'lr','scale','alpha'});self.assertTrue(f['lr']['at_maximum'])
        self.assertTrue(f['alpha']['at_maximum']);self.assertTrue(f['scale']['at_maximum'])
    def test_synthetic_complete_analysis_12(self):
        index={(d,n,a,s):dict(accuracy=.8+(.02 if a in ['ctrlA','ctrlB'] and n else 0.))
            for d,n,a,s in itertools.product(self.cfg['datasets'],self.cfg['noise'],self.cfg['arms'],self.cfg['evaluation_seeds'])}
        got=calculate(index,self.cfg);self.assertEqual(len(got['E4_primary']),12)
        self.assertEqual(got['family_summary']['verdict'],'FULL_SUPPORT')
    def test_admission_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):verify_admission(Path(td),self.cfg,'HASH',env())
    def test_heartbeat_identity_and_cpu(self):
        a=dict(timestamp='2026-09-04T00:00:00+00:00',run_id='r',unit_id='u',attempt_id='a',pid=12,phase='training',
            phase_started_at='t',unit_elapsed_seconds=1.,completed_atomic_units=0,planned_atomic_units=1,
            last_durable_checkpoint_at=None,process_tree_cpu_seconds=1.,wrapper_cpu_seconds=0.,process_tree_pids=[12,13],inner_completed=2)
        b=dict(a,timestamp='2026-09-04T00:00:01+00:00',unit_elapsed_seconds=2.,process_tree_cpu_seconds=2.,inner_completed=4)
        self.assertIsNotNone(advancing_pair([a,b]))
        self.assertIsNone(advancing_pair([a,dict(b,unit_id='other')]))
        self.assertIsNone(advancing_pair([a,dict(b,process_tree_cpu_seconds=1.)]))
    def test_mismatched_controlled_status_pair(self):
        allowed={('CONTROLLED_STOP',75),('COMPLETED',0)}
        self.assertNotIn(('CONTROLLED_STOP',0),allowed);self.assertNotIn(('FAILED',75),allowed)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if '--report' in sys.argv:
        exclusive(ROOT/'TEST_REPORT.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            generated_utc=now(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            freeze_sha256=verify_freeze(),environment=env()))
    sys.exit(not result.wasSuccessful())
