from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import load_config, matrix, write_json


def main():
    parser = argparse.ArgumentParser(description="Reproducible Burgers research pipeline")
    parser.add_argument("command",choices=["doctor","verify-solver","generate-data","validate-data","matrix",
        "train","evaluate","summarize","build-report","benchmark","reuse-data"])
    parser.add_argument("--config",default="configs/study.yaml")
    parser.add_argument("--work-dir",default="artifacts/study")
    parser.add_argument("--device",choices=["auto","cpu","mps","cuda"])
    parser.add_argument("--microbatch",type=int)
    parser.add_argument("--run",help="Exact registered run name; omit to run the complete matrix")
    parser.add_argument("--source-data-root",help="Validated source work directory for reuse-data")
    parser.add_argument("--data-config",help="Separate evaluation-only distribution configuration")
    parser.add_argument("--data-root",help="Separate validated evaluation dataset work directory")
    parser.add_argument("--analysis-root",help="Separate evaluation/summary/report output directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.microbatch:
        if args.microbatch < 1:
            parser.error("microbatch must be positive")
        cfg["runtime"]["microbatch"] = args.microbatch
    root = Path(args.work_dir)
    if args.command == "reuse-data" and not args.source_data_root:
        parser.error("reuse-data requires --source-data-root")
    if bool(args.data_config) != bool(args.data_root):
        parser.error("--data-config and --data-root must be supplied together")
    if args.data_config and (args.command != "evaluate" or not args.analysis_root):
        parser.error("Shift evaluation requires evaluate and a separate --analysis-root")
    root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(root.resolve()/".matplotlib"))
    specs = list(matrix(cfg))
    if args.run:
        specs = [s for s in specs if s["name"] == args.run]
        if not specs:
            parser.error("Unknown run; use the matrix command to list valid names")
    if args.command == "matrix":
        write_json(root/"experiment-registry.json",{"config":cfg,"runs":list(matrix(cfg))})
        print(json.dumps(specs,indent=2)); return
    if args.command == "reuse-data":
        from .data import reuse_dataset
        reuse_dataset(cfg,root,args.source_data_root)
        print(f"Dataset attached: {root}"); return
    if args.command in ("doctor","benchmark"):
        from .hardware import doctor, benchmark
        result = doctor(cfg,root) if args.command == "doctor" else benchmark(cfg,root)
    elif args.command == "verify-solver":
        from .numerics import verify_solver
        result = verify_solver(cfg,root)
    elif args.command in ("generate-data","validate-data"):
        from .data import generate,validate
        result = generate(cfg,root) if args.command == "generate-data" else validate(cfg,root)
        print(f"{args.command} complete: {root}"); return
    elif args.command == "train":
        from .hardware import doctor
        from .training import train
        doctor(cfg,root)
        write_json(root/"experiment-registry.json",{"config":cfg,"runs":list(matrix(cfg))})
        for spec in specs:
            train(cfg,root,spec)
        return
    elif args.command == "evaluate":
        from .evaluation import evaluate,evaluate_controls
        options = {"data_cfg":load_config(args.data_config) if args.data_config else None,
                   "data_root":args.data_root,"output_root":args.analysis_root}
        for spec in specs:
            evaluate(cfg,root,spec,**options)
        evaluate_controls(cfg,root,**options)
        return
    elif args.command == "summarize":
        from .reporting import summarize
        summarize(cfg,root,args.analysis_root)
        print(f"Summary written: {Path(args.analysis_root or root)/'summary'}"); return
    else:
        from .reporting import build_report
        print(build_report(cfg,root,args.analysis_root)); return
    print(json.dumps(result,indent=2,default=str))


if __name__ == "__main__":
    main()
