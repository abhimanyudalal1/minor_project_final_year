from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from .common import choose_device, environment, gradients_finite, matrix, seed_all, synchronize, write_json
from .models import build_model, parameter_count
from .numerics import initial_condition, project, solve
from .physics import training_loss
from .training import predict
from .storage import storage_budget


def doctor(cfg,root):
    device = choose_device(cfg["runtime"]["device"])
    seed_all(0)
    results = []
    for spec in benchmark_cases(cfg):
        kind = spec["model"]
        cpu = build_model(kind,cfg,spec)
        target = copy.deepcopy(cpu).to(device)
        x = torch.randn(2,1,cfg["data"]["base_n"])
        y = torch.randn_like(x)
        def run(model,x,y):
            out = model(x)
            total,_,_ = training_loss(model,x,y,{"mean":0.0,"std":1.0},cfg["data"]["dt"],cfg["data"]["viscosity"],spec)
            total.backward()
            grads = [p.grad.detach().cpu() for p in model.parameters()]
            torch.optim.Adam(model.parameters(),lr=1e-3).step()
            return out.detach().cpu(),grads
        a,ga = run(cpu,x,y)
        b,gb = run(target,x.to(device),y.to(device))
        prediction_ok = torch.allclose(a,b,rtol=1e-4,atol=1e-5)
        gradients_ok = all(torch.allclose(a,b,rtol=1e-3,atol=1e-5) for a,b in zip(ga,gb))
        results.append({"model":kind,"variant":spec["name"],"parameters":parameter_count(cpu),"prediction_passed":prediction_ok,
                        "gradients_passed":gradients_ok,"max_prediction_difference":float((a-b).abs().max())})
    path = Path(root)
    path.mkdir(parents=True,exist_ok=True)
    budget = storage_budget(cfg,path)
    report = {"passed":all(r["prediction_passed"] and r["gradients_passed"] for r in results),
        "device":str(device),"mps_available":torch.backends.mps.is_available(),"cuda_available":torch.cuda.is_available(),
        "free_disk_gb":budget["free_bytes"]/1e9,"required_disk_gb":budget["required_free_bytes"]/1e9,
        "storage_budget":budget,"storage_passed":budget["free_bytes"] >= budget["required_free_bytes"],
        "checks":results,"environment":environment()}
    write_json(path/"verification"/f"doctor-{device.type}.json",report)
    if not report["passed"]:
        raise RuntimeError("Backend parity check failed; use a passing backend before training")
    return report


def benchmark_cases(cfg):
    # One representative per distinct variant; include the pushforward graph.
    return [s for s in matrix(cfg) if s["seed"] == cfg["training"]["seeds"][0]]


def benchmark(cfg,root,updates=10):
    device = choose_device(cfg["runtime"]["device"])
    seed_all(999)
    d = cfg["data"]
    start = time.perf_counter()
    trajectory = solve(initial_condition(d["reference_n"],999999,d),d["viscosity"],d["dt"],d["final_time"],cfg["solver"]["initial_dt"])
    generation_seconds = time.perf_counter()-start
    low = project(trajectory,d["base_n"]).astype(np.float32)
    norm = {"mean":float(low.mean()),"std":float(low.std())}
    rows = []
    for spec in benchmark_cases(cfg):
        kind, weight, method = spec["model"], spec["weight"], spec["derivative"]
        model = build_model(kind,cfg,spec).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"])
        samples = np.arange(cfg["training"]["batch_size"]) % (len(low)-1)
        x = torch.as_tensor(low[samples,None,:])
        y = torch.as_tensor(low[samples+1,None,:])
        timings = []
        for update in range(updates+1):  # First update is an excluded warmup.
            synchronize(device); before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(x),cfg["runtime"]["microbatch"]):
                a = x[offset:offset+cfg["runtime"]["microbatch"]].to(device)
                b = y[offset:offset+cfg["runtime"]["microbatch"]].to(device)
                total,supervised,pde = training_loss(model,a,b,norm,d["dt"],d["viscosity"],spec)
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite benchmark loss")
                (total*len(a)/len(x)).backward()
                float(supervised); float(pde)  # Include the training loop's logging synchronizations.
            if not gradients_finite(model):
                raise FloatingPointError("Nonfinite benchmark gradient")
            optimizer.step(); synchronize(device)
            if update:
                timings.append(time.perf_counter()-before)
        model.eval()
        synchronize(device); before = time.perf_counter()
        with torch.no_grad():
            current = x[:cfg["runtime"]["evaluation_batch"]].to(device)
            for _ in range(round(d["train_time"]/d["dt"])):
                current = predict(model,current,norm)
        synchronize(device)
        validation_batch_seconds = time.perf_counter()-before
        steps = math.ceil(cfg["training"].get("train_trajectories", d["counts"]["train"])*round(d["train_time"]/d["dt"])/cfg["training"]["batch_size"])
        train_seconds = np.mean(timings)*steps*cfg["training"]["epochs"]
        validation_seconds = validation_batch_seconds*math.ceil(d["counts"]["validation"]/cfg["runtime"]["evaluation_batch"])*math.ceil(cfg["training"]["epochs"]/cfg["training"]["validation_every"])
        rows.append({"model":kind,"variant":spec.get("family",spec["name"].rsplit("-s",1)[0]),"weight":weight,"derivative":method,"effective_update_seconds":float(np.mean(timings)),
            "measured_updates":updates,"update_seconds_std":float(np.std(timings)),
            "validation_batch_seconds":validation_batch_seconds,"estimated_training_validation_hours_per_run":float((train_seconds+validation_seconds)/3600)})
    total_hours = 0
    for spec in matrix(cfg):
        row = next(r for r in rows if r["variant"] == spec.get("family",spec["name"].rsplit("-s",1)[0]))
        total_hours += row["estimated_training_validation_hours_per_run"]
    result = {"purpose":"Synthetic workload benchmark, not a reduced research dataset", "device":str(device),
        "generation_seconds_per_trajectory":generation_seconds,
        "estimated_serial_generation_hours":generation_seconds*sum(d["counts"].values())/3600,
        "estimated_full_training_validation_hours":total_hours,"cases":rows,"environment":environment(),
        "limitations":"Short benchmark excludes disk I/O, checkpointing, final evaluation, solver timestep refinement, and sustained thermal throttling. Estimates are not guarantees."}
    write_json(Path(root)/"verification"/"benchmark.json",result)
    return result
