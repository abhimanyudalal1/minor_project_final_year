"""Serial, resumable v2 workflow. Run status for inspection; run prepares and trains.

The primary workflow reuses validated data without regenerating or changing it.
The stress workflow generates independent evaluation-only data and never trains
on it. Source and configurations are frozen before invoking the public CLI.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from production import freeze_source


PROJECT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, state):
    state['updated_utc'] = now()
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(path)


def execute(args):
    root = Path(args.work_dir).resolve()
    execution = root/'execution'
    if args.action == 'status':
        path = execution/('stress-status.json' if args.stress_status else 'status.json')
        print(path.read_text() if path.exists() else 'No revision workflow recorded')
        return
    execution.mkdir(parents=True, exist_ok=True)
    with open(execution/'worker.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = freeze_source(execution, Path(args.config).resolve())
        config = source/'configs/production.yaml'
        env = dict(os.environ, PYTHONPATH=str(source/'src'), PYTHONUNBUFFERED='1')
        env.setdefault('MPLCONFIGDIR', str(root/'.matplotlib'))
        cfg = json.loads(subprocess.check_output([sys.executable, '-c',
            'import json,sys; from burgers_research.common import load_config; print(json.dumps(load_config(sys.argv[1])))',
            str(config)], env=env, cwd=PROJECT, text=True))
        if 'variants' not in cfg['training']:
            raise ValueError('This runner requires a dedicated explicit-variant configuration')
        commands = []

        def command(stage, cfg_path=config, work=root, extra=()):
            return [sys.executable, '-m', 'burgers_research.cli', stage, '--config', str(cfg_path),
                    '--work-dir', str(work), '--device', args.device, *map(str, extra)]

        stress = args.action == 'stress'
        if stress:
            # Check selection readiness before paying for data generation.
            subprocess.run([sys.executable, '-c',
                'import sys; from burgers_research.common import load_config; from burgers_research.training import lock_selection; lock_selection(load_config(sys.argv[1]),sys.argv[2])',
                str(config), str(root)], check=True, env=env, cwd=PROJECT)
            stress_config = execution/'stress-config.yaml'
            requested = Path(args.stress_config).read_bytes()
            if stress_config.exists() and stress_config.read_bytes() != requested:
                raise ValueError('Stress configuration changed; use a separate experiment directory')
            if not stress_config.exists():
                stress_config.write_bytes(requested)
            data_root = Path(args.stress_data_root).resolve()
            output = root/'stress-low-viscosity'
            for stage in ('verify-solver', 'generate-data', 'validate-data'):
                commands.append((stage, command(stage, stress_config, data_root)))
            commands.append(('evaluate', command('evaluate', extra=(
                '--data-config', stress_config, '--data-root', data_root, '--analysis-root', output))))
            for stage in ('summarize', 'build-report'):
                commands.append((stage, command(stage, extra=('--analysis-root', output))))
        else:
            if args.generate_fixture:
                if not cfg['fixture']:
                    raise ValueError('--generate-fixture requires fixture: true')
                for stage in ('verify-solver', 'generate-data', 'validate-data'):
                    commands.append((stage, command(stage)))
            else:
                commands.append(('reuse-data', command('reuse-data', extra=('--source-data-root', Path(args.source_data_root).resolve()))))
                commands.append(('validate-data', command('validate-data')))
            commands.extend((stage, command(stage)) for stage in ('matrix', 'doctor'))
            if args.action == 'run':
                commands.extend((stage, command(stage)) for stage in ('train', 'evaluate', 'summarize', 'build-report'))

        path = execution/('stress-status.json' if stress else 'status.json')
        state = json.loads(path.read_text()) if path.exists() else {'completed_stages': {}, 'started_utc': now()}
        state.update(status='running', pid=os.getpid(), source_snapshot=str(source), child_pid=None)
        child = None

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        try:
            for stage, cmd in commands:
                signature = hashlib.sha256(json.dumps(cmd).encode()).hexdigest()
                previous = state['completed_stages'].get(stage)
                if previous and previous != signature:
                    raise ValueError(f'Completed stage inputs changed: {stage}')
                if previous:
                    continue
                log_path = execution/f'{"stress-" if stress else ""}{stage}.log'
                state.update(stage=stage, command=cmd, log=str(log_path), heartbeat_utc=now())
                write_status(path, state)
                with open(log_path, 'ab', buffering=0) as log:
                    child = subprocess.Popen(cmd, env=env, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                    state['child_pid'] = child.pid
                    while True:
                        try:
                            code = child.wait(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            state['heartbeat_utc'] = now(); write_status(path, state)
                    if code:
                        raise RuntimeError(f'{stage} failed with exit {code}; inspect {log_path}')
                state['completed_stages'][stage] = signature
                state['child_pid'] = None
                write_status(path, state)
            state.update(status='prepared' if args.action == 'prepare' else 'complete', stage='finished')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', message=str(exc), child_pid=None)
            raise
        finally:
            write_status(path, state)
        print(json.dumps({'status': state['status'], 'status_file': str(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'stress', 'status'])
    parser.add_argument('--config', default='configs/revision.yaml')
    parser.add_argument('--work-dir', default='artifacts/revision')
    parser.add_argument('--source-data-root', default='artifacts/study')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--stress-config', default='configs/shock-evaluation.yaml')
    parser.add_argument('--stress-data-root', default='artifacts/shock-evaluation')
    parser.add_argument('--stress-status', action='store_true')
    parser.add_argument('--generate-fixture', action='store_true')
    execute(parser.parse_args())
