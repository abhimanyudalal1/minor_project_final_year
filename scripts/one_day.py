"""Run the fixed custom-data studies and stress evaluation within 24 hours.

One GPU queue and one CPU data-generation queue; no hyperparameter search.
All child processes use a frozen source snapshot. A deadline preserves completed
artifacts and records incomplete work rather than weakening selection checks.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from production import freeze_source

PROJECT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def stop_child(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def progress(base):
    result = {}
    for name in ('lowdata', 'full'):
        root = base/name
        item = {'completed_training_runs': len(list((root/'runs').glob('*/result.json')))}
        path = root/'execution/status.json'
        if path.exists():
            state = json.loads(path.read_text())
            item.update(stage=state.get('stage'), status=state.get('status'), heartbeat_utc=state.get('heartbeat_utc'))
        log = root/'execution/train.log'
        if log.exists():
            with open(log, 'rb') as f:
                f.seek(max(0, log.stat().st_size-4096))
                lines = f.read().decode(errors='replace').splitlines()
            item['latest_training_line'] = lines[-1] if lines else ''
        result[name] = item
    return result


def write_review(base, state):
    lines = ['# One-day custom-data experiment', '',
             f"Updated: {now()}. Status: **{state['status']}**.",
             f"Deadline: {state['deadline_utc']} (UTC).", '',
             'Fixed hyperparameters; four variants, three seeds, 60 epochs. '
             'Two training budgets: 200 and 2,000 trajectories. '
             'No PDEBench data or hyperparameter search.', '',
             '## Execution', '', '| Task | Status |', '|---|---|']
    for name, task in state['tasks'].items():
        lines.append(f"| {name} | {task['status']} |")
    lines += ['', '## Test results available so far', '',
              'Numbers below are mean relative L2 errors across seeds and trajectories. '
              'The 200-step rollout begins at t=0; extrapolation averages only t>1. '
              'Incomplete or failed groups are not treated as successful results.', '',
              '| Training data | Evaluation | Model | N | Steps | Rollout error | Seed SD | Failure rate |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    reports = []
    for name in ('lowdata', 'full'):
        for kind, directory in [('smooth', base/name), ('low-viscosity shift', base/name/'stress-low-viscosity')]:
            path = directory/'summary/summary.json'
            if not path.exists():
                continue
            doc = json.loads(path.read_text())
            for row in doc['table']:
                if row['metric'] != 'rollout_error' or row['adapter'] != 'native':
                    continue
                def fmt(v):
                    return 'FAILED / unavailable' if v is None else f'{v:.6g}'
                lines.append(f"| {name} | {kind} | {row['family']} | {row['resolution']} | {row['horizon']} | {fmt(row['mean'])} | {fmt(row['std'])} | {row['failure_rate']:.2%} |")
            report = directory/'report/report.md'
            if report.exists():
                reports.append(f'- [{name}: {kind}]({report.relative_to(base)})')
    lines += ['', '## Detailed reports', '', *reports, '',
              'Interpretation must include seed variability, failure rates, interpolation '
              'and numerical-reference controls. Mean conservation in the projected model '
              'is an enforced property, not evidence of accurate dynamics. The low-viscosity '
              'set changes the physical parameter of models trained at viscosity 0.01. '
              'It is not an established external benchmark.', '']
    (base/'ONE_DAY_RESULTS.md').write_text('\n'.join(lines))


def worker(args):
    base = Path(args.work_dir).resolve()
    execution = base/'execution'
    execution.mkdir(parents=True, exist_ok=True)
    status_path = base/'status.json'
    with open(execution/'worker.lock', 'a') as lock, open(PROJECT/'artifacts/study/execution/worker.lock', 'a') as accelerator_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(accelerator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = freeze_source(execution, base/'full.yaml')
        env = dict(os.environ, PYTHONPATH=str(source/'src'), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(base/'.matplotlib'))
        # Freeze all three configs, including the separate subset and stress protocols.
        hashes = {}
        for name in ('full', 'lowdata', 'shock'):
            payload = (base/f'{name}.yaml').read_bytes()
            frozen = execution/f'{name}.yaml'
            if frozen.exists() and frozen.read_bytes() != payload:
                raise ValueError(f'{name} configuration changed after launch')
            frozen.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()

        def cli(stage, cfg, root, extra=()):
            return [sys.executable, '-m', 'burgers_research.cli', stage, '--config', str(cfg),
                    '--work-dir', str(root), '--device', 'mps', *map(str, extra)]

        tasks = {}
        cpu = []
        gpu = []

        def add(name, queue, cmd, dependencies=()):
            queue.append(name)
            tasks[name] = {'command': cmd, 'dependencies': list(dependencies), 'status': 'pending'}

        shock_root = base/'shock-data'
        for stage in ('verify-solver', 'generate-data', 'validate-data'):
            name = f'shock-{stage}'
            add(name, cpu, cli(stage, execution/'shock.yaml', shock_root), cpu[-1:])
        for name in ('lowdata', 'full'):
            add(f'{name}-study', gpu, [sys.executable, str(source/'scripts/run_revision.py'), 'run',
                '--config', str(execution/f'{name}.yaml'), '--work-dir', str(base/name),
                '--source-data-root', str(PROJECT/'artifacts/study'), '--device', 'mps'])
        for name in ('lowdata', 'full'):
            root = base/name
            output = root/'stress-low-viscosity'
            cfg = root/'execution/source/configs/production.yaml'
            add(f'{name}-shift-evaluate', gpu, cli('evaluate', cfg, root, (
                '--data-config', execution/'shock.yaml', '--data-root', shock_root, '--analysis-root', output)),
                (f'{name}-study', 'shock-validate-data'))
            for stage in ('summarize', 'build-report'):
                previous = gpu[-1]
                add(f'{name}-shift-{stage}', gpu, cli(stage, cfg, root, ('--analysis-root', output)), (previous,))

        if status_path.exists():
            state = json.loads(status_path.read_text())
            if state['config_hashes'] != hashes:
                raise ValueError('One-day configuration hashes changed')
            for name, task in tasks.items():
                old = state['tasks'][name]
                if task['command'] != old['command']:
                    raise ValueError('One-day task command changed')
                if old['status'] == 'complete':
                    task.update(old)
        else:
            state = {'started_utc': now(), 'deadline_utc': args.deadline_utc, 'config_hashes': hashes}
        deadline = datetime.fromisoformat(state['deadline_utc']).timestamp()
        if deadline <= time.time():
            raise ValueError('The original one-day deadline has expired')
        state.update(status='running', pid=os.getpid(), source_snapshot=str(source), tasks=tasks)
        save(status_path, state)
        write_review(base, state)
        running = {'cpu': None, 'gpu': None}
        awake = None
        if sys.platform == 'darwin':
            awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while True:
                if time.time() >= deadline:
                    state['status'] = 'deadline_reached'
                    break
                for lane, queue in (('cpu', cpu), ('gpu', gpu)):
                    active = running[lane]
                    if active:
                        name, child, log = active
                        code = child.poll()
                        if code is None:
                            continue
                        log.close()
                        tasks[name].update(status='complete' if code == 0 else 'failed', exit_code=code, finished_utc=now())
                        running[lane] = None
                    # Failed tasks block only their dependants. Independent study proceeds.
                    for name in queue:
                        task = tasks[name]
                        if task['status'] != 'pending':
                            continue
                        deps = [tasks[d]['status'] for d in task['dependencies']]
                        if any(s in ('failed', 'blocked') for s in deps):
                            task['status'] = 'blocked'
                            continue
                        if not all(s == 'complete' for s in deps):
                            continue
                        log_path = execution/f'{name}.log'
                        log = open(log_path, 'ab', buffering=0)
                        child = subprocess.Popen(task['command'], cwd=PROJECT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                        task.update(status='running', pid=child.pid, started_utc=now(), log=str(log_path))
                        running[lane] = (name, child, log)
                        break
                state['heartbeat_utc'] = now()
                state['remaining_hours'] = max(0, (deadline-time.time())/3600)
                state['progress'] = progress(base)
                save(status_path, state)
                if all(t['status'] in ('complete', 'failed', 'blocked') for t in tasks.values()):
                    state['status'] = 'complete' if all(t['status'] == 'complete' for t in tasks.values()) else 'finished_with_failures'
                    break
                time.sleep(5)
        except BaseException as exc:
            state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', message=str(exc))
            raise
        finally:
            for active in running.values():
                if active:
                    name, child, log = active
                    stop_child(child); log.close()
                    if tasks[name]['status'] == 'running':
                        tasks[name].update(status='interrupted', finished_utc=now())
            if awake is not None:
                awake.terminate()
            state['updated_utc'] = now()
            state['progress'] = progress(base)
            save(status_path, state)
            write_review(base, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'worker', 'status', 'stop'])
    parser.add_argument('--work-dir', default='artifacts/one-day')
    parser.add_argument('--deadline-utc', default=None)
    args = parser.parse_args()
    if args.deadline_utc is None:
        args.deadline_utc = (datetime.now(timezone.utc)+timedelta(hours=24)).isoformat()
    base = Path(args.work_dir).resolve()
    if args.action == 'worker':
        worker(args); return
    path = base/'status.json'
    if args.action == 'status':
        print(path.read_text() if path.exists() else 'Not started'); return
    (base/'execution').mkdir(parents=True, exist_ok=True)
    with open(base/'execution/worker.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if args.action == 'stop':
                os.kill(json.loads(path.read_text())['pid'], signal.SIGTERM)
                print('Stop requested; checkpoints retained'); return
            raise SystemExit('One-day worker is already running')
        if args.action == 'stop':
            print('No active one-day worker'); return
    with open(base/'execution/launcher.log', 'ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
            '--work-dir', str(base), '--deadline-utc', args.deadline_utc], cwd=PROJECT,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({'pid':child.pid, 'status_file':str(path), 'deadline_utc':args.deadline_utc}))


if __name__ == '__main__':
    main()
