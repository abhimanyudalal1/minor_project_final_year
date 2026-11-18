"""Matched, validation-only 0.001 versus 0.0003 learning-rate study."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from burgers_research.common import load_config, read_json, write_json
import budget_run

PROJECT = Path(__file__).resolve().parents[1]
STUDY = PROJECT/'artifacts/study'
ROOT = STUDY/'learning-rate-0003'
OUTPUT = ROOT/'analysis-epoch26'
TARGET_EPOCH = 26


def initialize():
    ROOT.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    for name in ('data','verification'):
        link=ROOT/name
        if not link.exists():
            link.symlink_to((STUDY/name).resolve(),target_is_directory=True)
    cfg=load_config(STUDY/'execution/source/configs/production.yaml')
    cfg['training']['learning_rate']=0.0003
    cfg['runtime']['device']='mps'
    cfg['runtime']['microbatch']=64
    cfg['runtime']['evaluation_batch']=32
    config_path=ROOT/'config.yaml'
    if config_path.exists():
        old=yaml.safe_load(config_path.read_text())
        assert {k:v for k,v in old.items() if k!='runtime'} == {k:v for k,v in cfg.items() if k!='runtime'}, 'Study configuration changed'
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    budget_run.ROOT=ROOT
    budget_run.OUT=OUTPUT
    return cfg


def compare():
    control=read_json(STUDY/'eight-hour/results.json')
    candidate=read_json(OUTPUT/'results.json')
    assert control['epoch']==candidate['epoch']==TARGET_EPOCH
    assert control['dataset_hash']==candidate['dataset_hash']
    lines=['# Matched learning-rate comparison','',
           'Full dataset; three paired seeds; identical architectures and 26 epochs. '
           'Validation only, N=256, 100-step mean relative L2. Lower is better. '
           'The 0.001 control consists of previously completed matched epoch-26 checkpoints.', '',
           '| Architecture | LR 0.001 mean ± seed SD | LR 0.0003 mean ± seed SD | Paired differences (0.0003 − 0.001) |',
           '|---|---:|---:|---|']
    for family in ('unet','fno','physics-fno'):
        a={r['seed']:r['rollout_error'] for r in control['runs'] if r['family']==family}
        b={r['seed']:r['rollout_error'] for r in candidate['runs'] if r['family']==family}
        assert set(a)==set(b)=={0,1,2}
        differences=[b[i]-a[i] for i in range(3)]
        av=list(a.values());bv=list(b.values())
        lines.append(f"| {family} | {np.mean(av):.5f} ± {np.std(av,ddof=1):.5f} | {np.mean(bv):.5f} ± {np.std(bv,ddof=1):.5f} | {', '.join(f'{x:+.5f}' for x in differences)} |")
    lines += ['', 'Results compare endpoints, not selected best epochs. Microbatch 32 was used in the control; '
              'the candidate used 64 for most of the first 10 epochs, 128 for part of seed 0, and 64 thereafter; '
              'both use the same effective batch and example order, but floating-point reductions may differ. '
              'This early screen does not establish convergence; three seeds are insufficient for strong statistical conclusions. '
              'The same validation set informed the decision to investigate learning rate; these are exploratory findings. '
              'No test data were accessed. Full per-seed results and mass-drift values are in analysis/results.md.', '']
    (OUTPUT/'comparison.md').write_text('\n'.join(lines))


def controller(cfg):
    status_path=OUTPUT/'status.json'
    state={'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
           'learning_rate':0.0003,'control_learning_rate':0.001,'target_epoch':TARGET_EPOCH,'completed_runs':[],
           'microbatch':64,'training_cache':'full base-grid training trajectories in RAM'}
    child=None
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        with open(STUDY/'execution/worker.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for spec in budget_run.selected(cfg):
                name=spec['name']
                state.update(stage='training',current_run=name,heartbeat_utc=datetime.now(timezone.utc).isoformat())
                write_json(status_path,state)
                milestone=OUTPUT/'milestones'/name/f'epoch-{TARGET_EPOCH}.pt'
                if not milestone.exists():
                    with open(OUTPUT/'training.log','ab',buffering=0) as log:
                        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'train','--run',name],
                            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                        while child.poll() is None:
                            state['heartbeat_utc']=datetime.now(timezone.utc).isoformat()
                            write_json(status_path,state)
                            time.sleep(10)
                        if child.returncode:
                            raise RuntimeError(f'Training failed for {name}; inspect training.log')
                state['completed_runs'].append(name)
                write_json(status_path,state)
            state.update(stage='validation',current_run=None)
            write_json(status_path,state)
            budget_run.summary(cfg,budget_run.selected(cfg))
            compare()
            state.update(status='complete',stage='finished')
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
        state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',message=str(exc))
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(status_path,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['controller','train'])
    parser.add_argument('--run')
    args=parser.parse_args()
    cfg=initialize()
    if args.mode=='train':
        original_store=budget_run.training.Store
        class CachedStore(original_store):
            def __init__(self,cfg,root,split,validated=True):
                super().__init__(cfg,root,split,validated)
                self.cache=None
                if split=='train':
                    self.cache=np.stack([self.trajectory(i,self.base_n,self.steps+1) for i in range(len(self.rows))])
                    print(f'Cached complete training grid: {self.cache.nbytes/1e6:.1f} MB',flush=True)
            def pairs(self,indices):
                if self.cache is None:
                    return super().pairs(indices)
                indices=np.asarray(indices); trajectory=indices//self.steps;step=indices%self.steps
                return self.cache[trajectory,step,None,:],self.cache[trajectory,step+1,None,:]
        budget_run.training.Store=CachedStore
        budget_run.train_to(cfg,next(s for s in budget_run.selected(cfg) if s['name']==args.run),TARGET_EPOCH)
    else:
        controller(cfg)
