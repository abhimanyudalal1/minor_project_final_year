"""Run a separate variant study using the existing validated PDEBench archive."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from production import freeze_source
from burgers_research.common import data_signature, file_hash, load_config, read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def attach(cfg, root, source):
    manifest = read_json(source/'data/manifest.json')
    check = read_json(source/'verification/data.json')
    if 'PDEBench' not in manifest.get('source', {}).get('dataset', ''):
        raise ValueError('Expected an imported PDEBench archive')
    if manifest['data_signature'] != data_signature(cfg):
        raise ValueError('Dataset configuration mismatch')
    if not check['passed'] or check['manifest_sha256'] != file_hash(source/'data/manifest.json'):
        raise ValueError('Source verification mismatch')
    for shard in manifest['shards']:
        if file_hash(source/'data'/shard['file']) != shard['sha256']:
            raise ValueError(f"Source shard checksum mismatch: {shard['file']}")
    target = root/'data'
    if target.exists() or target.is_symlink():
        if target.resolve() != (source/'data').resolve():
            raise ValueError('Refusing to replace another dataset')
    else:
        target.symlink_to(source/'data', target_is_directory=True)
    (root/'verification').mkdir(exist_ok=True)
    shutil.copy2(source/'verification/data.json', root/'verification/data.json')
    write_json(root/'verification/reused-data.json', {
        'passed': True, 'source': str(source), 'dataset_hash': manifest['dataset_hash'],
        'checked_shards': len(manifest['shards']), 'checked_utc': now(),
        'method': 'Verified imported manifest and every converted shard checksum; original conversion validation retained.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pdebench-unet-constrained.yaml')
    parser.add_argument('--work-dir', default='artifacts/pdebench-unet-constrained')
    parser.add_argument('--source-data-root', default='artifacts/pdebench')
    parser.add_argument('--prepare-source', help='Verified HDF5 archive to convert if source-data-root is not prepared')
    parser.add_argument('--wait-for-status', help='Prior workflow must complete before GPU work starts')
    parser.add_argument('--runtime-hours', type=float, help='Maximum hours after GPU stages begin; checkpoints retained on timeout')
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    execution = root/'execution'
    execution.mkdir(parents=True, exist_ok=True)
    with open(execution/'worker.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = freeze_source(execution, Path(args.config).resolve())
        config = snapshot/'configs/production.yaml'
        cfg = load_config(config)
        path = execution/'status.json'
        state = read_json(path) if path.exists() else {'started_utc': now(), 'completed_stages': []}
        state.update(status='running', pid=os.getpid(), child_pid=None, message='', source_snapshot=str(snapshot))
        child = None

        def save():
            state.update(updated_utc=now(), heartbeat_utc=now())
            write_json(path, state)

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        env = dict(os.environ, PYTHONPATH=str(snapshot/'src'), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(root/'.matplotlib'))
        try:
            source = Path(args.source_data_root).resolve()
            if args.prepare_source and not (source/'verification/data.json').exists():
                state.update(stage='convert-data'); save()
                cmd = [sys.executable, str(snapshot/'scripts/prepare_pdebench.py'),
                       '--source', str(Path(args.prepare_source).resolve()), '--root', str(source), '--config', str(config)]
                with open(execution/'conversion.log', 'ab', buffering=0) as log:
                    child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state['child_pid'] = child.pid; save()
                    while True:
                        try:
                            code = child.wait(timeout=10); break
                        except subprocess.TimeoutExpired:
                            save()
                    if code:
                        raise RuntimeError(f'Dataset conversion failed with exit {code}; inspect conversion.log')
                state['child_pid'] = None
            # Recheck data on resume as well; imported data has no local solver certificate.
            state.update(stage='attach-verified-data'); save()
            attach(cfg, root, Path(args.source_data_root).resolve())
            if 'attach-verified-data' not in state['completed_stages']:
                state['completed_stages'].append('attach-verified-data')
            for stage in ['matrix', 'doctor', 'train', 'evaluate', 'summarize', 'build-report']:
                if stage in state['completed_stages']:
                    continue
                if stage == 'doctor' and args.wait_for_status:
                    state.update(stage='waiting-for-prior-study', waiting_for=str(Path(args.wait_for_status).resolve())); save()
                    while True:
                        prior = read_json(args.wait_for_status)
                        if prior['status'] == 'complete':
                            break
                        if prior['status'] in ('failed', 'stopped', 'paused', 'deadline_reached'):
                            raise RuntimeError('Prior study stopped; inspect it before resuming the GPU queue')
                        time.sleep(10); save()
                if stage == 'doctor' and args.runtime_hours and 'deadline_utc' not in state:
                    deadline = time.time() + args.runtime_hours * 3600
                    state['deadline_utc'] = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
                deadline = datetime.fromisoformat(state['deadline_utc']).timestamp() if state.get('deadline_utc') else float('inf')
                if time.time() >= deadline:
                    raise TimeoutError('Runtime budget reached; resume from retained checkpoints with an extended deadline')
                cmd = [sys.executable, '-m', 'burgers_research.cli', stage, '--config', str(config),
                       '--work-dir', str(root), '--device', 'mps']
                logpath = execution/f'{stage}.log'
                state.update(stage=stage, stage_started_utc=now(), command=cmd, log=str(logpath)); save()
                with open(logpath, 'ab', buffering=0) as log:
                    child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state['child_pid'] = child.pid; save()
                    while True:
                        try:
                            code = child.wait(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            if time.time() >= deadline:
                                raise TimeoutError('Runtime budget reached; checkpoints retained')
                            save()
                    if code:
                        raise RuntimeError(f'{stage} failed with exit {code}; inspect {logpath}')
                state['completed_stages'].append(stage)
                state['child_pid'] = None; save()
            state.update(status='complete', stage='finished')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'deadline_reached' if isinstance(exc, TimeoutError) else 'failed',
                         message=str(exc), child_pid=None)
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
