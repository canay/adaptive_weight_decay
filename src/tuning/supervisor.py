"""MC009 bounded two-phase durable supervisor; no failed-run automatic resume."""
import argparse,concurrent.futures,shutil,signal,threading,time,traceback
from core import *
from engine import run_cell,validate_payload
from selection import lock_selection,evaluation
from liveness import sample_process_tree,append_durable
from admission import verify_admission

def output_bytes(out):
    total=0
    for p in out.rglob('*'):
        try:
            if p.is_file() and p.suffix!='.tmp':total+=p.stat().st_size
        except FileNotFoundError:pass
    return total

def run(args):
    cfg=load(ROOT/'config.json');frozen=verify_freeze();environment=env();smoke=args.phase in ['smoke','repeat'];calibration=args.phase=='calibration'
    if args.phase=='main':
        verify_admission(ROOT,cfg,frozen,environment)
    if args.stop_after and (args.phase!='smoke' or args.stop_after!=2):
        raise RuntimeError('CONTROLLED_STOP_ONLY_TWO_UNIT_SMOKE')
    out=ROOT/'outputs'/args.phase;out.mkdir(parents=True,exist_ok=True)
    terminal=out/'terminal_status.json';resume_count=0
    if terminal.exists():
        old=load(terminal)
        if (old.get('status'),old.get('exit_code')) not in [('CONTROLLED_STOP',75),('COMPLETED',0)]:
            raise RuntimeError('FAILED_PREDECESSOR_REQUIRES_NEW_REPAIR')
        if old.get('environment')!=environment or old.get('freeze_sha256')!=frozen:
            raise RuntimeError('PREDECESSOR_SOURCE_ENV_DRIFT')
    lock=ROOT/f'{args.phase}.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    attempt=uuid.uuid4().hex;t0=time.monotonic();stop=threading.Event();err=[];active={};submitted={};mtx=threading.RLock()
    previous_pids=set();observed_processes={};resumed_files={};completed_files={}
    planned=90 if calibration else len(development(cfg,True))+20 if smoke else cfg['planned_units']
    elapsed_predecessors=sum(load(p)['elapsed_seconds'] for p in out.glob('terminal_*.json') if p.name!='terminal_status.json')
    state=dict(status='RUNNING',run_id=cfg['run_id'],phase=args.phase,attempt_id=attempt,pid=os.getpid(),
        started_utc=now(),freeze_sha256=frozen,environment=environment,planned_units=planned,completed_units=0,
        current_stage='development',phase_started_at=now(),last_durable_checkpoint_at=None)
    def sample():
        nonlocal previous_pids
        with mtx:
            process,previous_pids=sample_process_tree(previous_pids)
            for p in psutil.Process().children(recursive=True):
                try:observed_processes[p.pid]=p.create_time()
                except psutil.Error:pass
            event=dict(state,timestamp=now(),elapsed_seconds=time.monotonic()-t0,
                phase_elapsed_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(state['phase_started_at'])).total_seconds(),
                unit_id='supervisor_'+state['current_stage'],unit_elapsed_seconds=time.monotonic()-t0,
                completed_atomic_units=state['completed_units'],planned_atomic_units=planned,**process,
                active_units=[unit_key(c) for c in tuple(active.values())],
                free_bytes=shutil.disk_usage(ROOT).free,available_ram_bytes=psutil.virtual_memory().available)
            atomic(out/'heartbeat.json',event)
            append_durable(out/f'heartbeat_{attempt}.jsonl',event)
    def monitor():
        try:
            while not stop.wait(cfg['heartbeat_seconds']):sample()
        except BaseException as e:err.append(repr(e))
    th=threading.Thread(target=monitor,daemon=True);pool=None;code=0;new=0
    def terminated(signum,frame):raise KeyboardInterrupt('SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,terminated)
    def resources():
        if err:raise RuntimeError('SUPERVISOR_HEARTBEAT_FAILED '+str(err))
        if elapsed_predecessors+time.monotonic()-t0>cfg['watchdog_seconds']:raise TimeoutError('RUN_WATCHDOG')
        if any(time.monotonic()-t>cfg['cell_timeout_seconds']+30 for t in submitted.values()):raise TimeoutError('OPAQUE_CELL_TIMEOUT')
        if shutil.disk_usage(ROOT).free<cfg['disk_floor_bytes']:raise RuntimeError('DISK_FLOOR')
        if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('RAM_FLOOR')
    try:
        if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('START_RAM_FLOOR')
        resources();sample();th.start();last_budget_check=0
        for stage in (['development'] if calibration else ['development','evaluation']):
            state.update(current_stage=stage,phase_started_at=now());selection_sha=None
            if stage=='development':plan=calibration_plan(cfg) if calibration else development(cfg,smoke)
            else:
                selection_sha,selected=lock_selection(out,cfg,frozen,environment,smoke)
                plan=evaluation(cfg,selected,smoke);state['selection_sha256']=selection_sha
            todo=[]
            for c in plan:
                path=out/'cells'/(unit_key(c)+'.json')
                if validate_payload(path,c,frozen,environment,selection_sha):
                    resume_count+=1;resumed_files[path.name]=sha(path)
                    append_durable(out/f'resume_{attempt}.jsonl',dict(status='SKIPPED_VALIDATED',unit_id=unit_key(c),
                        sha256=sha(path),attempt_id=attempt,timestamp=now()))
                else:todo.append(c)
            if stage=='development' and terminal.exists() and load(terminal)['status']=='CONTROLLED_STOP':
                stopped=load(out/'STOPPED_TWO_HASHES.json')
                if stopped['files']!=resumed_files or stopped['attempt_id']!=load(terminal)['attempt_id']:
                    raise RuntimeError('CONTROLLED_PREFIX_DRIFT')
            state['completed_units']=resume_count+new
            if args.stop_after:todo=todo[:max(0,args.stop_after-new)]
            pool=concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']);queue=iter(todo)
            while True:
                while len(active)<cfg['workers']:
                    c=next(queue,None)
                    if c is None:break
                    f=pool.submit(run_cell,c,str(out),frozen,cfg,environment,selection_sha)
                    with mtx:active[f]=c;submitted[f]=time.monotonic()
                if not active:break
                done,_=concurrent.futures.wait(active,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    key,cached=f.result()
                    with mtx:del active[f];del submitted[f]
                    new+=not cached;state['completed_units']=resume_count+new
                    path=out/'cells'/(key+'.json');completed_files[path.name]=sha(path)
                    state['last_durable_checkpoint_at']=now()
                    atomic(out/'progress.json',dict(state,timestamp=now(),last_completed_unit=key));sample()
                    print(f"{state['completed_units']}/{planned} {key}",flush=True)
                resources()
                if time.monotonic()-last_budget_check>15:
                    if output_bytes(out)>cfg['run_output_budget_bytes']:raise RuntimeError('OUTPUT_BUDGET')
                    last_budget_check=time.monotonic()
            pool.shutdown();pool=None
            if args.stop_after and new>=args.stop_after:
                expected={unit_key(c)+'.json' for c in development(cfg,True)[:2]}
                if set(completed_files)!=expected or {p.name for p in (out/'cells').glob('*.json')}!=expected:
                    raise RuntimeError('CONTROLLED_PREFIX_NOT_EXACT')
                still_alive=[]
                for pid,created in observed_processes.items():
                    try:
                        if psutil.Process(pid).create_time()==created:still_alive.append(pid)
                    except psutil.NoSuchProcess:pass
                if still_alive:raise RuntimeError('CONTROLLED_CHILDREN_ALIVE '+str(still_alive))
                exclusive(out/'STOPPED_TWO_HASHES.json',dict(attempt_id=attempt,timestamp=now(),
                    freeze_sha256=frozen,files=completed_files,observed_processes=observed_processes,
                    descendants_alive_after_shutdown=still_alive))
                state['status']='CONTROLLED_STOP';code=75;break
        else:
            if state['completed_units']!=planned:raise RuntimeError('COUNT_INCOMPLETE')
            state['status']='COMPLETED'
    except BaseException as e:
        code=130 if isinstance(e,KeyboardInterrupt) else 1
        state.update(status='TERMINATED' if code==130 else 'FAILED',error=repr(e));traceback.print_exc()
    finally:
        if pool is not None:
            children=psutil.Process().children(recursive=True)
            for p in children:
                try:p.terminate()
                except psutil.Error:pass
            _,alive=psutil.wait_procs(children,timeout=5)
            for p in alive:
                try:p.kill()
                except psutil.Error:pass
            pool.shutdown(wait=False,cancel_futures=True)
        stop.set()
        if th.is_alive():th.join(timeout=2)
        state.update(exit_code=code,completed_utc=now(),elapsed_seconds=time.monotonic()-t0,resumed_units=resume_count,new_units=new,
            completed_unit_ids=sorted(set(completed_files)|set(resumed_files)),resumed_files=resumed_files)
        atomic(terminal,state);exclusive(out/f'terminal_{attempt}.json',state)
        sample();lock.unlink()
    return code

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','repeat','calibration','main'],required=True)
    ap.add_argument('--stop-after',type=int,default=0)
    sys.exit(run(ap.parse_args()))
