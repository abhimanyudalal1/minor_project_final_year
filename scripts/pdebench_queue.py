"""Prepare PDEBench while custom study runs; serialize GPU work and stop at 06:00 IST."""
from datetime import datetime,timezone
from pathlib import Path
import json,os,signal,subprocess,sys,time
ROOT=Path('artifacts/pdebench');ROOT.mkdir(parents=True,exist_ok=True)
STATUS=ROOT/'queue-status.json'; DEADLINE=datetime.fromisoformat('2026-11-17T00:30:00+00:00').timestamp()
def save(s):
 s['updated_utc']=datetime.now(timezone.utc).isoformat(); p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATUS)
def main():
 state=json.loads(STATUS.read_text()) if STATUS.exists() else {'started_utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':'2026-11-17T00:30:00+00:00','completed_stages':[]}
 state.update(status='running',pid=os.getpid(),child_pid=None);save(state)
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR=str((ROOT/'.matplotlib').resolve()))
 child=None
 def stop(signum,frame): raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop)
 signal.signal(signal.SIGINT,stop)
 try:
  tasks=[('conversion',[sys.executable,'scripts/prepare_pdebench.py']),('doctor',[sys.executable,'-m','burgers_research.cli','doctor','--config','configs/pdebench.yaml','--work-dir',str(ROOT),'--device','mps']),('train',[sys.executable,'-m','burgers_research.cli','train','--config','configs/pdebench.yaml','--work-dir',str(ROOT),'--device','mps']),('evaluate',[sys.executable,'-m','burgers_research.cli','evaluate','--config','configs/pdebench.yaml','--work-dir',str(ROOT),'--device','mps']),('summarize',[sys.executable,'-m','burgers_research.cli','summarize','--config','configs/pdebench.yaml','--work-dir',str(ROOT),'--device','mps']),('build-report',[sys.executable,'-m','burgers_research.cli','build-report','--config','configs/pdebench.yaml','--work-dir',str(ROOT),'--device','mps'])]
  for stage,cmd in tasks:
   if stage in state.get('completed_stages',[]): continue
   if stage=='doctor':
    state['stage']='waiting_for_custom_gpu';save(state)
    while True:
     p=Path('artifacts/eight-run/full/execution/status.json');s=json.loads(p.read_text())
     if s['status']=='complete':break
     if s['status'] in ('failed','stopped'):raise RuntimeError('Custom study stopped or failed; GPU queue requires investigation')
     if time.time()>=DEADLINE:state['status']='deadline_reached';return
     time.sleep(15)
   if time.time()>=DEADLINE:state['status']='deadline_reached';return
   state.update(stage=stage,command=cmd);save(state)
   with open(ROOT/f'{stage}.log','ab',buffering=0) as log:
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    state['child_pid']=child.pid;save(state)
    while child.poll() is None:
     if time.time()>=DEADLINE:
      os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);state.update(status='deadline_reached',child_pid=None);return
     time.sleep(5);state['heartbeat_utc']=datetime.now(timezone.utc).isoformat();save(state)
    if child.returncode:raise RuntimeError(f'{stage} failed with exit {child.returncode}; inspect {stage}.log')
    state.setdefault('completed_stages',[]).append(stage);state['child_pid']=None;save(state)
  state.update(status='complete',stage='finished')
 except BaseException as e:
  if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
  state.update(status='paused' if isinstance(e,KeyboardInterrupt) else 'failed',message=str(e),child_pid=None)
  if not isinstance(e,KeyboardInterrupt): raise
 finally:save(state)
if __name__=='__main__':main()
