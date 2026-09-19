import copy, tempfile, unittest
from pathlib import Path
import numpy as np
from runner import *
from analyze import contrast, holm
from strong_baseline_fidelity import self_test

class TestProtocol(unittest.TestCase):
    def test_constant_training_feature_varying_test(self):
        train=np.array([[.0097,0.],[.0097,2.],[.0097,4.]])
        test=np.array([[10.,1.],[-5.,3.]])
        x,xt,mu,scale,constant=standardize_train_only(train,test)
        self.assertEqual(constant.tolist(),[True,False])
        self.assertEqual(scale[0],1.)
        self.assertLess(float(np.abs(xt).max()),11.)
        old=((test-train.mean(0))/(train.std(0)+1e-8)).astype(np.float32)
        self.assertGreater(float(np.abs(old[:,0]).max()),1e8)
        np.testing.assert_array_equal(xt[:,1],old[:,1])
        np.testing.assert_array_equal(x[:,1],((train-train.mean(0))/(train.std(0)+1e-8)).astype(np.float32)[:,1])
    def test_test_values_never_fit_scaling(self):
        train=np.random.RandomState(2).normal(size=(30,5));train[:,0]=.0097
        a=standardize_train_only(train,np.zeros((3,5)))
        b=standardize_train_only(train,np.full((7,5),1e9))
        for i in [0,2,3,4]:np.testing.assert_array_equal(a[i],b[i])
    def test_all_input_contexts_safe(self):
        from input_qa import inspect_inputs
        report=inspect_inputs()
        self.assertEqual(report['contexts'],90)
        self.assertEqual(report['status'],'PASS')
    def test_output_budget_ignores_temporary_siblings(self):
        with tempfile.TemporaryDirectory(prefix='f06-budget-') as tmp:
            root=Path(tmp)
            (root/'cell.json').write_bytes(b'12345')
            (root/'cell.json.unique.tmp').write_bytes(b'x'*200)
            self.assertEqual(output_bytes(root),5)
    def test_concurrent_atomic_writers(self):
        from unittest.mock import patch
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier, Lock
        import runner
        barrier=Barrier(2); replacement_lock=Lock(); real_replace=os.replace
        def simultaneous_replace(source,target):
            barrier.wait(timeout=10)
            with replacement_lock:return real_replace(source,target)
        with tempfile.TemporaryDirectory(prefix='f06-thread-atomic-') as tmp:
            target=Path(tmp)/'heartbeat.json'
            with patch.object(runner.os,'replace',simultaneous_replace):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(atomic,target,dict(writer=i)) for i in range(2)]
                    for future in futures:future.result()
            self.assertIn(load(target)['writer'],[0,1])
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
    def test_plan(self):
        cfg=load(ROOT/'config.json');a,b=cells(cfg)
        self.assertEqual((len(a),len(b)),(1440,1080))
        self.assertEqual(len(set(key(c) for c in a+b)),2520)
    def test_pairing(self):
        cell=dict(dataset='digits',size='full',noise=.2,seed=10)
        full,nc,m=data_for(cell);half,_,_=data_for(dict(cell,size='half'))
        clean,_,_=data_for(dict(cell,noise=0.))
        np.testing.assert_array_equal(full[1][:len(half[1])],half[1])
        np.testing.assert_array_equal(full[0],clean[0])
        np.testing.assert_array_equal(full[4],clean[1])
        self.assertGreater(m['realized_noise'],0)
    def test_exact_replay(self):
        cfg=load(ROOT/'config.json');frozen=verify_bundle();env=envelope()
        for arm in ['ctrlA','ctrlB']:
            cell=dict(dataset='digits',size='half',noise=.2,seed=900,windows=5,
                      smoke=True,arm=arm,control='online',donor=None)
            with tempfile.TemporaryDirectory(prefix='f06-fidelity-') as tmp:
                run_cell(cell,tmp,frozen,cfg,env)
                donor=load(Path(tmp)/'cells'/(key(cell)+'.json'))
                other=dict(cell,control='replay',donor=900)
                run_cell(other,tmp,frozen,cfg,env)
                replay=load(Path(tmp)/'cells'/(key(other)+'.json'))
                self.assertEqual(donor['parameter_sha256'],replay['parameter_sha256'])
                self.assertEqual(donor['row']['test_acc'],replay['row']['test_acc'])
                for mode in ['reverse','constant']:
                    c=dict(other,control=mode);run_cell(c,tmp,frozen,cfg,env)
                    p=load(Path(tmp)/'cells'/(key(c)+'.json'))
                    self.assertAlmostEqual(donor['negative_log_total_shrinkage'],p['negative_log_total_shrinkage'],places=12)
                damaged=copy.deepcopy(donor);damaged['row']['test_acc']+=.01
                path=Path(tmp)/'cells'/(key(cell)+'.json');atomic(path,damaged)
                with self.assertRaises(RuntimeError):valid(path,cell,frozen,env)
    def test_statistics(self):
        cfg=load(ROOT/'config.json')
        self.assertEqual(contrast([0.]*9,cfg)['p_raw'],1.)
        self.assertEqual(contrast([1.]*9,cfg)['p_raw'],2/512)
        rows=holm([dict(p_raw=.01,mean=.01),dict(p_raw=.04,mean=.01)],cfg)
        self.assertEqual([r['p_holm'] for r in rows],[.02,.04])
    def test_baselines(self):
        self.assertEqual(set(self_test()),{'adamw','awd','swd','cwd'})
    def test_terminal_admission(self):
        self.assertTrue(admissible_predecessor(dict(status='CONTROLLED_STOP',exit_code=75)))
        self.assertFalse(admissible_predecessor(dict(status='FAILED',exit_code=75)))
        self.assertFalse(admissible_predecessor(dict(status='CONTROLLED_STOP',exit_code=1)))
        self.assertFalse(admissible_predecessor(dict(status='TERMINATED',exit_code=130)))
    def test_common_initialization_and_batches(self):
        cfg=load(ROOT/'config.json');frozen=verify_bundle();env=envelope();ids=[]
        with tempfile.TemporaryDirectory(prefix='f06-pairing-') as tmp:
            for noise in [0.,.2]:
                for arm in cfg['arms']:
                    cell=dict(dataset='digits',size='half',noise=noise,seed=950,windows=3,
                              smoke=True,arm=arm,control='online',donor=None)
                    run_cell(cell,tmp,frozen,cfg,env)
                    p=load(Path(tmp)/'cells'/(key(cell)+'.json'))
                    ids.append((p['initial_parameters_sha256'],p['minibatches_sha256']))
        self.assertEqual(len(set(ids)),1)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestProtocol)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    atomic(ROOT/'logs/unit_test_receipt.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        freeze_sha256=verify_bundle(),environment=envelope(),created_utc=now()))
    sys.exit(0 if result.wasSuccessful() else 1)
