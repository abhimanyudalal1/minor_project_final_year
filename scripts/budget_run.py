"""Eight-hour validation pilot using the unchanged frozen scientific implementation."""
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
from burgers_research.common import load_config, matrix, read_json, write_json
from burgers_research.data import Store
from burgers_research.models import build_model
from burgers_research.training import predict
import burgers_research.training as training

ROOT = Path(__file__).resolve().parents[1]/"artifacts/study"
OUT = ROOT/"eight-hour"
CFG = ROOT/"execution/source/configs/production.yaml"


def selected(cfg):
    return [s for s in matrix(cfg) if s['seed'] in (0,1,2) and
            (s['weight']==0 or (s['weight']==1 and s['derivative']=='spectral'))]


class EpochReached(Exception):
    pass


def train_to(cfg, spec, target):
    """Stop after an existing epoch-boundary save, never mid-accumulation."""
    original = training.atomic_save
    def save(path, checkpoint):
        original(path, checkpoint)
        state=checkpoint['state']
        if Path(path).name=='last.pt' and state['epoch'] >= target and state['cursor']==0:
            original(OUT/'milestones'/spec['name']/f"epoch-{target}.pt", checkpoint)
            raise EpochReached
    training.atomic_save=save
    try:
        training.train(cfg, ROOT, spec)
    except EpochReached:
        pass
    finally:
        training.atomic_save=original


def epochs(name):
    return {int(p.stem.split('-')[1]) for p in (OUT/'milestones'/name).glob('epoch-*.pt')}


@torch.no_grad()
def summary(cfg, specs):
    common=set.intersection(*(epochs(s['name']) for s in specs))
    if not common:
        write_json(OUT/'results.json',{'status':'incomplete','reason':'No epoch completed by all nine runs',
                                     'completed_epochs':{s['name']:sorted(epochs(s['name'])) for s in specs}})
        return
    epoch=max(common)
    store=Store(cfg,ROOT,'validation')
    norm=store.manifest['normalization']
    rows=[]
    try:
        for spec in specs:
            checkpoint=torch.load(OUT/'milestones'/spec['name']/f'epoch-{epoch}.pt',map_location='cpu',weights_only=False)
            model=build_model(spec['model'],cfg).to('mps').eval()
            model.load_state_dict(checkpoint['model'])
            records=[]
            for start in range(0,len(store.rows),16):
                indices=list(range(start,min(start+16,len(store.rows))))
                truth=torch.tensor(np.stack([store.trajectory(i,256,101) for i in indices]),device='mps')
                cur=truth[:,0:1]
                scale=truth[:,0].square().mean(-1).sqrt().clamp_min(1e-12)
                mass0=truth[:,0].mean(-1)
                errors=[]; masses=[]
                failed=torch.zeros(len(indices),dtype=torch.bool,device='mps')
                for t in range(1,101):
                    cur=predict(model,cur,norm)
                    rms=cur[:,0].square().mean(-1).sqrt()
                    failed |= ~torch.isfinite(cur).all(dim=(1,2)) | (rms>100*scale)
                    denom=torch.maximum(truth[:,t].square().mean(-1).sqrt(),(1e-8*scale).clamp_min(1e-12))
                    err=(cur[:,0]-truth[:,t]).square().mean(-1).sqrt()/denom
                    errors.append(torch.where(failed,float('inf'),err))
                    masses.append((cur[:,0].mean(-1)-mass0).abs()/scale)
                error=torch.stack(errors).mean(0).cpu().tolist()
                drift=torch.stack(masses).max(0).values.cpu().tolist()
                for j,index in enumerate(indices):
                    records.append({'id':store.rows[index][2],'rollout_error':error[j],'normalized_max_mass_drift':drift[j]})
            row={'name':spec['name'],'seed':spec['seed'],'family':spec['model'] if not spec['weight'] else 'physics-fno',
                 'epoch':epoch,'rollout_error':float(np.mean([r['rollout_error'] for r in records])),
                 'normalized_max_mass_drift':float(np.mean([r['normalized_max_mass_drift'] for r in records]))}
            rows.append(row)
            write_json(OUT/f"validation-{spec['name']}.json",{'summary':row,'records':records})
        groups=[]
        for family in ('unet','fno','physics-fno'):
            values=[r['rollout_error'] for r in rows if r['family']==family]
            groups.append({'family':family,'mean':float(np.mean(values)),
                           'seed_sd':float(np.std(values,ddof=1)) if np.isfinite(values).all() else None})
        write_json(OUT/'results.json',{'epoch':epoch,'split':'validation','resolution':256,'horizon':100,
                                      'runs':rows,'groups':groups,'dataset_hash':store.manifest['dataset_hash']})
        lines=['# Eight-hour pilot results','',f'All nine runs compared at epoch {epoch}, on all 250 validation trajectories.',
               'The full dataset is unchanged. Test data remain locked. These are early-training results, not convergence claims.','',
               '| Model | Mean rollout relative L2 | Seed SD |','|---|---:|---:|']
        lines += [f"| {g['family']} | {g['mean']:.6g} | {g['seed_sd']} |" for g in groups]
        lines += ['', '## Per-seed results','', '| Run | Rollout error | Normalized maximum mass drift |','|---|---:|---:|']
        lines += [f"| {r['name']} | {r['rollout_error']:.6g} | {r['normalized_max_mass_drift']:.6g} |" for r in rows]
        lines += ['', '## Next steps','',
                  'Inspect the learning curves and seed spread before extending training. If the ranking varies across seeds, prioritize replication. '
                  'If losses are still fluctuating, run an equally budgeted, validation-only learning-rate study before architecture changes. '
                  'Then compare resolution transfer and energy/residual diagnostics. Preserve unfavorable results; do not open the test set until the final protocol is fixed.']
        (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    finally:
        store.close()


def controller(cfg, hours):
    OUT.mkdir(exist_ok=True)
    with open(ROOT/'execution/worker.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time(); deadline=start+hours*3600
        # Reserve the last 15 minutes for matched-epoch validation and reporting.
        training_end=deadline-min(900,hours*3600/4)
        specs=selected(cfg)
        state={'status':'running','pid':os.getpid(),'started_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
               'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
               'training_ends_utc':datetime.fromtimestamp(training_end,timezone.utc).isoformat(),
               'runs':[s['name'] for s in specs]}
        child=None
        def stopped(signum,frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stopped)
        try:
            # Preserve exact epoch-10 models from the existing seed-0 pilot.
            for spec in specs:
                path=ROOT/'runs'/spec['name']/'best.pt'
                if path.exists():
                    c=torch.load(path,map_location='cpu',weights_only=False)
                    e=c['state']['epoch']
                    if c['state']['cursor']==0 and not (OUT/'milestones'/spec['name']/f'epoch-{e}.pt').exists():
                        training.atomic_save(OUT/'milestones'/spec['name']/f'epoch-{e}.pt',c)
            while time.time()<training_end:
                completed={s['name']:max(epochs(s['name']),default=0) for s in specs}
                minimum=min(completed.values())
                spec=next(s for s in specs if completed[s['name']]==minimum)
                target=10 if minimum<10 else (20 if minimum<20 else minimum+1)
                state.update(stage='training',current_run=spec['name'],target_epoch=target,completed_epochs=completed)
                with open(OUT/'training.log','ab',buffering=0) as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'train','--run',spec['name'],
                        '--epoch',str(target)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    while child.poll() is None and time.time()<training_end:
                        state['heartbeat_utc']=datetime.now(timezone.utc).isoformat()
                        write_json(OUT/'status.json',state)
                        time.sleep(min(10,max(0,training_end-time.time())))
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=30)
                        break
                    if child.returncode:
                        raise RuntimeError(f"Training failed for {spec['name']}; see training.log")
            state.update(stage='validation',heartbeat_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT/'status.json',state)
            with open(OUT/'validation.log','ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'summary'],stdout=log,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                try:
                    code=child.wait(timeout=max(1,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=30)
                    raise RuntimeError('Eight-hour limit reached during validation; saved training remains available')
                if code:
                    raise RuntimeError('Validation failed; see validation.log')
            state.update(status='complete',stage='finished')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=30)
            state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',message=str(exc))
            raise
        finally:
            state['updated_utc']=datetime.now(timezone.utc).isoformat()
            write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['controller','train','summary'])
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--run'); parser.add_argument('--epoch',type=int)
    args=parser.parse_args()
    cfg=load_config(CFG); cfg['runtime']['device']='mps'
    if args.mode=='train':
        train_to(cfg,next(s for s in selected(cfg) if s['name']==args.run),args.epoch)
    elif args.mode=='summary':
        summary(cfg,selected(cfg))
    else:
        controller(cfg,args.hours)
