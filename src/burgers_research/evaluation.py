from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline
import torch

from .common import choose_device, digest, experiment_signature, file_hash, read_json, source_hash, synchronize, write_json
from .data import Store
from .models import build_model
from .numerics import derivative, project
from .physics import residual
from .training import lock_selection, predict


def interpolate(u,n,kind="cubic"):
    old = u.shape[-1]
    if old == n:
        return u.copy()
    closed = np.concatenate((u,u[...,:1]),axis=-1)
    x = np.arange(old+1)/old
    target = np.arange(n)/n
    if kind == "cubic":
        return CubicSpline(x,closed,axis=-1,bc_type="periodic")(target).astype(u.dtype)
    if kind == "linear":
        index = target*old
        left = np.floor(index).astype(int)
        fraction = index-left
        return ((1-fraction)*closed[...,left]+fraction*closed[...,left+1]).astype(u.dtype)
    raise ValueError(kind)


@torch.no_grad()
def step(model,u,norm,device,base_n,adapter="native"):
    n = u.shape[-1]
    a = project(u,base_n) if adapter != "native" else u
    value = predict(model,torch.as_tensor(a[:,None,:],device=device,dtype=torch.float32),norm)
    out = value[:,0].cpu().numpy()
    return interpolate(out,n,adapter) if adapter != "native" else out


def diagnostics(pred,truth,cfg):
    """Per-trajectory metrics; arrays are [time,space], all calculations in float64."""
    pred,truth = pred.astype(np.float64),truth.astype(np.float64)
    d,e = cfg["data"],cfg["evaluation"]
    rms0 = max(float(np.sqrt(np.mean(truth[0]**2))),1e-12)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        norms = np.sqrt(np.mean(truth**2,axis=-1))
        denom = np.maximum(norms,max(e["denominator_floor"]*rms0,1e-12))
        error = np.sqrt(np.mean((pred-truth)**2,axis=-1))/denom
        failed = (~np.isfinite(pred).all(-1)) | (np.sqrt(np.mean(pred**2,axis=-1)) > e["divergence_factor"]*rms0)
        failed = np.maximum.accumulate(failed)
        error[failed] = np.inf
        mass = pred.mean(-1)-pred[0].mean()
        energy = 0.5*np.mean(pred**2,axis=-1)
        truth_energy = 0.5*np.mean(truth**2,axis=-1)
        dissipation = d["viscosity"]*np.mean(derivative(pred,1)**2,axis=-1)
        integral = np.concatenate(([0.0],np.cumsum((dissipation[1:]+dissipation[:-1])*d["dt"]/2)))
        balance = energy-energy[0]+integral
        residuals = {}
        for method in ("spectral","finite_difference"):
            a,b = torch.from_numpy(pred[:-1]),torch.from_numpy(pred[1:])
            r = residual(a,b,d["dt"],d["viscosity"],method).numpy()
            residuals[method] = np.sqrt(np.mean(r*r,axis=-1)).tolist()
    horizons = {}
    for h in e["horizons"]:
        if h >= len(pred):
            continue
        horizons[str(h)] = {"failed":bool(failed[h]),"rollout_error":float(error[1:h+1].mean()),
            "terminal_error":float(error[h]),"max_mass_drift":float(np.max(np.abs(mass[:h+1]))),
            "normalized_mass_drift":float(np.max(np.abs(mass[:h+1]))/rms0),
            "energy_error":float(np.mean(np.abs(energy[1:h+1]-truth_energy[1:h+1]))),
            "energy_increase_count":int(np.sum(np.diff(energy[:h+1])>1e-7*max(energy[0],1e-24))),
            "dissipation_balance_error":float(np.max(np.abs(balance[:h+1]))),
            "spectral_residual":float(np.mean(residuals["spectral"][:h])),
            "finite_difference_residual":float(np.mean(residuals["finite_difference"][:h]))}
        train_steps = round(d["train_time"]/d["dt"])
        if h > train_steps:
            horizons[str(h)]["extrapolation_error"] = float(error[train_steps+1:h+1].mean())
    return {"horizons":horizons,"curves":{"relative_l2":error.tolist(),"mass_drift":mass.tolist(),
        "energy":energy.tolist(),"reference_energy":truth_energy.tolist(),"dissipation_balance":balance.tolist(),
        "spectral_residual":residuals["spectral"],"finite_difference_residual":residuals["finite_difference"]}}


def evaluation_context(cfg, root, data_cfg=None, data_root=None, output_root=None):
    """Lock model selection before opening any ID or shifted test Store."""
    import copy
    root = Path(root)
    selection = lock_selection(cfg, root)
    target_cfg = copy.deepcopy(data_cfg or cfg)
    target_cfg["evaluation"] = copy.deepcopy(cfg["evaluation"])
    target_cfg["runtime"] = copy.deepcopy(cfg["runtime"])
    if any(target_cfg["data"][k] != cfg["data"][k] for k in ("dt", "base_n", "train_time")):
        raise ValueError("Shift evaluation requires the trained timestep, base grid and training time window")
    if max(cfg["evaluation"]["horizons"])*target_cfg["data"]["dt"] > target_cfg["data"]["final_time"] + 1e-12:
        raise ValueError("Shift dataset is shorter than the requested evaluation")
    target_root = Path(data_root or root)
    output = Path(output_root or root)
    if data_cfg is not None and output.resolve() == root.resolve():
        raise ValueError("Shift evaluation needs a separate output directory")
    store = Store(target_cfg, target_root, "test")
    context = {"config": {k:v for k,v in target_cfg.items() if k != "runtime"},
               "training_dataset_hash": selection["dataset_hash"], "dataset_hash": store.manifest["dataset_hash"],
               "training_signature": experiment_signature(cfg), "shifted": data_cfg is not None,
               "source_hash": source_hash()}
    context["hash"] = digest(context)
    path = output/"evaluation-context.json"
    if path.exists() and read_json(path) != context:
        store.close()
        raise ValueError("Evaluation context changed; use a new output directory")
    write_json(path, context)
    return target_cfg, store, output, context


def evaluate(cfg,root,spec, *, data_cfg=None, data_root=None, output_root=None):
    root = Path(root)
    eval_cfg, store, output_base, context = evaluation_context(cfg,root,data_cfg,data_root,output_root)
    device = choose_device(cfg["runtime"]["device"])
    checkpoint_path = root/"runs"/spec["name"]/"best.pt"
    checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if checkpoint["dataset_hash"] != context["training_dataset_hash"] or checkpoint["signature"] != experiment_signature(cfg) or checkpoint["spec"] != spec:
        raise ValueError("Checkpoint/data mismatch")
    if checkpoint["source_hash"] != source_hash():
        raise ValueError("Source changed since training; evaluate with the recorded source version")
    model = build_model(spec["model"],cfg,spec).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    norm = checkpoint["normalization"]
    max_h = max(cfg["evaluation"]["horizons"])
    batch = cfg["runtime"]["evaluation_batch"]
    output = output_base/"evaluation"/spec["name"]
    try:
        for n in eval_cfg["data"]["resolutions"]:
            adapters = ["native"] if n == cfg["data"]["base_n"] else ["native","cubic","linear"]
            for adapter in adapters:
                path = output/f"n{n}-{adapter}.json"
                checkpoint_hash = file_hash(checkpoint_path)
                if path.exists():
                    existing = read_json(path)
                    if existing["checkpoint_hash"] != checkpoint_hash or existing["source_hash"] != source_hash() or existing.get("context_hash") != context["hash"]:
                        raise ValueError("Evaluation checkpoint changed")
                    continue
                records = []
                rollout_seconds,one_step_seconds = 0.0,0.0
                for start in range(0,len(store.rows),batch):
                    indices = list(range(start,min(start+batch,len(store.rows))))
                    truth = np.stack([store.trajectory(i,n,max_h+1) for i in indices])
                    prediction = np.full_like(truth,np.nan)
                    prediction[:,0] = truth[:,0]
                    initial_scale = np.maximum(np.sqrt(np.mean(truth[:,0]**2,axis=-1)),1e-12)
                    active = np.ones(len(indices),dtype=bool)
                    synchronize(device)
                    before = time.perf_counter()
                    for t in range(1,max_h+1):
                        if active.any():
                            prediction[active,t] = step(model,prediction[active,t-1],norm,device,cfg["data"]["base_n"],adapter)
                            valid = np.isfinite(prediction[:,t]).all(-1)
                            with np.errstate(over="ignore"):
                                valid &= np.sqrt(np.mean(prediction[:,t].astype(np.float64)**2,axis=-1)) <= cfg["evaluation"]["divergence_factor"]*initial_scale
                            active &= valid
                    synchronize(device)
                    rollout_seconds += time.perf_counter()-before
                    before = time.perf_counter()
                    teacher_mse = np.zeros(len(indices)); teacher_relative = np.zeros(len(indices))
                    for t in range(max_h):
                        next_step = step(model,truth[:,t],norm,device,cfg["data"]["base_n"],adapter)
                        mse = np.mean((next_step.astype(np.float64)-truth[:,t+1])**2,axis=-1)
                        denom = np.maximum(np.sqrt(np.mean(truth[:,t+1].astype(np.float64)**2,axis=-1)),
                            np.maximum(initial_scale*cfg["evaluation"]["denominator_floor"],1e-12))
                        teacher_mse += mse/max_h
                        teacher_relative += np.sqrt(mse)/denom/max_h
                    synchronize(device)
                    one_step_seconds += time.perf_counter()-before
                    for j,index in enumerate(indices):
                        records.append({"id":store.rows[index][2],"one_step_mse":teacher_mse[j],
                            "one_step_relative_l2":teacher_relative[j],**diagnostics(prediction[j],truth[j],eval_cfg)})
                write_json(path,{"spec":spec,"resolution":n,"adapter":adapter,"records":records,
                    "dataset_hash":store.manifest["dataset_hash"],"checkpoint_hash":checkpoint_hash,
                    "source_hash":source_hash(),"context_hash":context["hash"],
                    "training_dataset_hash":context["training_dataset_hash"],
                    "device":str(device),"batch_size":batch,"rollout_seconds":rollout_seconds,
                    "one_step_seconds":one_step_seconds,"fixture":cfg["fixture"]})
                print(f"Evaluated {spec['name']} N={n} {adapter}",flush=True)
    finally:
        store.close()


def evaluate_controls(cfg,root, *, data_cfg=None, data_root=None, output_root=None):
    root = Path(root)
    eval_cfg, store, output, context = evaluation_context(cfg,root,data_cfg,data_root,output_root)
    try:
        for n in eval_cfg["data"]["resolutions"]:
            for kind in ("reference","persistence","oracle_cubic","oracle_linear"):
                records = []
                for i,(_,_,tid) in enumerate(store.rows):
                    truth = store.trajectory(i,n)
                    if kind == "reference":
                        pred = truth
                    elif kind == "persistence":
                        pred = np.broadcast_to(truth[:1],truth.shape)
                    else:
                        coarse = store.trajectory(i,cfg["data"]["base_n"])
                        pred = interpolate(coarse,n,kind.split("_")[1])
                    records.append({"id":tid,**diagnostics(pred,truth,eval_cfg)})
                write_json(output/"evaluation"/"controls"/f"n{n}-{kind}.json",
                           {"resolution":n,"control":kind,"records":records,
                            "dataset_hash":store.manifest["dataset_hash"],"context_hash":context["hash"],"fixture":cfg["fixture"]})
    finally:
        store.close()
