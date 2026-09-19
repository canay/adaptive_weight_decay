"""Uniform durable heartbeat fields, including real descendant CPU accounting."""
import threading,time
from core import *

def sample_process_tree(previous_pids):
    parent=psutil.Process();tree=[];rss=0
    for p in [parent]+parent.children(recursive=True):
        try:
            cpu=sum(p.cpu_times()[:2]);rss+=p.memory_info().rss
            tree.append(dict(pid=p.pid,create_time=p.create_time(),cpu_seconds=cpu,rss_bytes=p.memory_info().rss))
        except psutil.Error:pass
    pids={p['pid'] for p in tree}
    return dict(process_tree=tree,process_tree_pids=sorted(pids),
        disappeared_pids=sorted(set(previous_pids)-pids),
        wrapper_cpu_seconds=sum(parent.cpu_times()[:2]),process_tree_cpu_seconds=sum(p['cpu_seconds'] for p in tree),
        process_cpu_seconds=sum(parent.cpu_times()[:2]),process_tree_rss_bytes=rss),pids

def append_durable(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('ab') as f:f.write(canonical(body)+b'\n');f.flush();os.fsync(f.fileno())

class CellHeartbeat:
    def __init__(self,out,cell,cfg):
        self.out=Path(out);self.t0=time.monotonic();self.phase_t0=self.t0
        self.lock=threading.RLock();self.stop=threading.Event();self.errors=[];self.pids=set()
        self.live=dict(timestamp=now(),run_id=cfg['run_id'],unit_id=unit_key(cell),attempt_id=uuid.uuid4().hex,
            pid=os.getpid(),phase='input',phase_started_at=now(),completed_steps=0,progress_denominator=0,
            completed_atomic_units=0,planned_atomic_units=1,last_durable_checkpoint_at=None)
        self.path=self.out/'worker_heartbeats'/(objhash(cell)[:24]+'__'+self.live['attempt_id']+'.jsonl')
        self.thread=threading.Thread(target=self.loop,args=(cfg['heartbeat_seconds'],),daemon=True)
        self.emit();self.thread.start()
    def phase(self,name):
        with self.lock:self.live.update(phase=name,phase_started_at=now());self.phase_t0=time.monotonic()
        self.emit()
    def emit(self):
        with self.lock:
            process,self.pids=sample_process_tree(self.pids)
            body=dict(self.live,**process,timestamp=now(),unit_elapsed_seconds=time.monotonic()-self.t0,
                phase_elapsed_seconds=time.monotonic()-self.phase_t0,inner_completed=self.live['completed_steps'],
                inner_total=self.live['progress_denominator'])
            atomic(self.out/'live'/f'{os.getpid()}.json',body);append_durable(self.path,body)
    def loop(self,cadence):
        try:
            while not self.stop.wait(cadence):self.emit()
        except BaseException as e:self.errors.append(repr(e))
    def close(self):
        self.stop.set();self.thread.join(timeout=2)
        if self.errors:raise RuntimeError('WORKER_HEARTBEAT_FAILED '+str(self.errors))
