"""Persistent, resumable stage runner. Only successful stages are marked complete."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPOSITORY = Path(__file__).resolve().parents[1]
STAGES = ("doctor", "matrix", "verify-solver", "generate-data", "validate-data",
          "train", "evaluate", "summarize", "build-report")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze_source(execution, config):
    snapshot = execution / "source"
    if snapshot.exists():
        manifest = json.loads((snapshot / "snapshot.json").read_text())
        for name, expected in manifest["files"].items():
            if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen source changed: {name}")
        if hashlib.sha256(config.read_bytes()).hexdigest() != manifest["files"]["configs/production.yaml"]:
            raise RuntimeError("Configuration changed; start a new study directory.")
        return snapshot
    temporary = execution / "source.partial"
    if temporary.exists():
        shutil.rmtree(temporary)  # Only this runner's incomplete snapshot.
    (temporary / "src").mkdir(parents=True)
    shutil.copytree(REPOSITORY/"src"/"burgers_research", temporary/"src"/"burgers_research",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (temporary/"configs").mkdir()
    shutil.copy2(config, temporary/"configs"/"production.yaml")
    for name in ("docs", "tests", "scripts"):
        shutil.copytree(REPOSITORY/name, temporary/name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("readme.md", "plan.md", "pyproject.toml", "requirements-macos.lock.txt", "PIPELINE_WALKTHROUGH.md"):
        shutil.copy2(REPOSITORY/name, temporary/name)
    files = {str(p.relative_to(temporary)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(temporary.rglob("*")) if p.is_file()}
    atomic_json(temporary/"snapshot.json", {"created":now(), "files":files})
    temporary.rename(snapshot)
    return snapshot


def worker(args):
    root = Path(args.work_dir).resolve()
    execution = root/"execution"
    execution.mkdir(parents=True,exist_ok=True)
    # The open descriptor owns the exclusive lock for the worker's lifetime.
    lock = open(execution/"worker.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = execution/"status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"completed_stages":[]}
    state.update(status="starting", pid=os.getpid(), started=now(), heartbeat=now(), device=args.device)
    atomic_json(status_path,state)
    stopped = False
    child = None
    awake = None

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    try:
        snapshot = freeze_source(execution,Path(args.config).resolve())
        state["source_snapshot"] = str(snapshot)
        env = dict(os.environ, PYTHONPATH=str(snapshot/"src"), PYTHONUNBUFFERED="1")
        env["MPLCONFIGDIR"] = str(execution/"matplotlib-cache")
        env["XDG_CACHE_HOME"] = str(execution/"cache")
        if sys.platform == "darwin" and Path("/usr/bin/caffeinate").exists():
            awake = subprocess.Popen(["/usr/bin/caffeinate","-i","-w",str(os.getpid())])
        for stage in STAGES:
            if stopped:
                raise InterruptedError("Stop requested")
            if stage in state["completed_stages"] and stage != "doctor":
                continue
            log_path = execution/f"{stage}.log"
            command = [sys.executable,"-m","burgers_research.cli",stage,"--config",
                       str(snapshot/"configs"/"production.yaml"),"--work-dir",str(root),"--device",args.device]
            state.update(status="running",stage=stage,stage_started=now(),log=str(log_path),command=command)
            with open(log_path,"ab",buffering=0) as log:
                log.write(f"\n--- Stage started {now()} ---\n".encode())
                child = subprocess.Popen(command,cwd=REPOSITORY,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL,start_new_session=True)
                state["child_pid"] = child.pid
                last_heartbeat = 0
                while child.poll() is None:
                    if stopped:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL)
                            child.wait()
                        raise InterruptedError("Stopped; rerun start to resume from saved artifacts")
                    if time.monotonic()-last_heartbeat >= 15:
                        state["heartbeat"] = now()
                        state["completed_shards"] = len(list((root/"data").glob("*.h5")))
                        state["finished_run_results"] = len(list((root/"runs").glob("*/result.json")))
                        atomic_json(status_path,state)
                        last_heartbeat = time.monotonic()
                    time.sleep(1)
                if child.returncode:
                    raise RuntimeError(f"{stage} failed with exit code {child.returncode}; inspect {log_path}")
            if stage not in state["completed_stages"]:
                state["completed_stages"].append(stage)
            state.update(heartbeat=now(),child_pid=None)
            atomic_json(status_path,state)
        state.update(status="complete",finished=now())
    except InterruptedError as exc:
        state.update(status="stopped",message=str(exc))
    except Exception as exc:
        state.update(status="failed",message=str(exc))
        raise
    finally:
        state.update(heartbeat=now(),child_pid=None)
        atomic_json(status_path,state)
        if awake is not None:
            awake.terminate()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["start","worker","status","stop"])
    parser.add_argument("--config",default=str(REPOSITORY/"configs"/"study.yaml"))
    parser.add_argument("--work-dir",default=str(REPOSITORY/"artifacts"/"study"))
    parser.add_argument("--device",choices=["mps","cuda","cpu"],default="mps")
    args = parser.parse_args()
    execution = Path(args.work_dir).resolve()/"execution"
    execution.mkdir(parents=True,exist_ok=True)
    status_path = execution/"status.json"
    if args.action == "worker":
        worker(args)
    elif args.action == "start":
        with open(execution/"worker.lock","a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Production worker is already running")
        with open(execution/"launcher.log","ab",buffering=0) as log:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker",
                "--config",str(Path(args.config).resolve()),"--work-dir",str(Path(args.work_dir).resolve()),
                "--device",args.device],cwd=REPOSITORY,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"worker_pid":process.pid,"status_file":str(status_path)}))
    elif args.action == "status":
        print(status_path.read_text() if status_path.exists() else "No production worker started")
    else:
        if not status_path.exists():
            raise SystemExit("No production worker started")
        state = json.loads(status_path.read_text())
        if state["status"] not in ("running","starting"):
            raise SystemExit(f"Worker is {state['status']}; no process was signalled")
        # A held lock confirms there is a live worker before signalling its recorded PID.
        with open(execution/"worker.lock","a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                os.kill(state["pid"],signal.SIGTERM)
                print("Stop requested; completed checkpoints and shards are retained")
            else:
                raise SystemExit("Worker is no longer alive; repeat start to resume")


if __name__ == "__main__":
    main()
