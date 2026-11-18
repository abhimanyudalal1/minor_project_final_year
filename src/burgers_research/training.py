from __future__ import annotations

import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .common import (choose_device, environment, experiment_signature, file_hash, gradients_finite, host_peak_memory_bytes, matrix, read_json,
                     seed_all, source_hash, synchronize, write_json)
from .data import Store
from .models import build_model, parameter_count
from .physics import training_loss


def predict(model,physical,norm):
    return model((physical-norm["mean"])/norm["std"])*norm["std"]+norm["mean"]


def atomic_save(path,state):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state,temporary)
    temporary.replace(path)


def rng_state():
    result = {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


@torch.no_grad()
def validation_score(model,store,cfg,device):
    model.eval()
    batch = cfg["runtime"]["evaluation_batch"]
    norm = store.normalization
    values = []
    for start in range(0,len(store.rows),batch):
        truth = torch.as_tensor(np.stack([store.trajectory(i,store.base_n,store.steps+1)
            for i in range(start,min(start+batch,len(store.rows)))]),device=device)
        current = truth[:,0:1]
        initial_rms = truth[:,0].square().mean(-1).sqrt()
        floor = (initial_rms*cfg["evaluation"]["denominator_floor"]).clamp_min(1e-12)
        errors = []
        for t in range(1,store.steps+1):
            current = predict(model,current,norm)
            error = (current[:,0]-truth[:,t]).square().mean(-1).sqrt()/torch.maximum(
                truth[:,t].square().mean(-1).sqrt(),floor)
            divergent = current[:,0].square().mean(-1).sqrt() > cfg["evaluation"]["divergence_factor"]*initial_rms.clamp_min(1e-12)
            error = torch.where(divergent | ~torch.isfinite(error),float("inf"),error)
            errors.append(error)
        values.extend(torch.stack(errors).mean(0).cpu().tolist())
    return float(np.mean(values))


def train(cfg,root,spec,max_updates=None):
    root = Path(root)
    folder = root/"runs"/spec["name"]
    folder.mkdir(parents=True,exist_ok=True)
    device = choose_device(cfg["runtime"]["device"])
    train_store = Store(cfg,root,"train")
    val_store = Store(cfg,root,"validation")
    norm = train_store.normalization
    signature = experiment_signature(cfg)
    seed_all(spec["seed"])
    model = build_model(spec["model"],cfg,spec).to(device)
    settings = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(),lr=settings["learning_rate"],weight_decay=0)
    scheduler = make_scheduler(optimizer, settings)
    state = {"epoch":0,"cursor":0,"updates":0,"best_score":float("inf"),"history":[],
             "epoch_sums":[0.0,0.0,0],"elapsed_seconds":0.0,"sampled_device_memory_bytes":0}
    env = environment()
    sessions = []
    if (folder/"last.pt").exists():
        # Only load locally generated, trusted checkpoints (optimizer/RNG require pickle).
        checkpoint = torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["dataset_hash"] != train_store.manifest["dataset_hash"] or checkpoint["spec"] != spec:
            raise ValueError("Checkpoint protocol/dataset/run mismatch")
        if checkpoint["source_hash"] != source_hash():
            raise ValueError("Source changed since checkpoint; use a new work directory for revised experiments.")
        if checkpoint.get("training_view_hash", train_store.view["hash"]) != train_store.view["hash"]:
            raise ValueError("Checkpoint training subset/normalization changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        state = checkpoint["state"]
        sessions = checkpoint["sessions"]
        # PyTorch's scheduler warning tracks calls only on this optimizer object.
        # A recovered end-of-epoch cursor can legitimately need scheduler.step()
        # before the next optimizer.step(); the saved updates already occurred.
        if state["updates"]:
            optimizer._opt_called = True
    sessions.append({"device":str(device),"runtime":cfg["runtime"],"environment":env})
    start_time = time.perf_counter()
    prior_elapsed = state["elapsed_seconds"]

    def save(path):
        synchronize(device)
        state["elapsed_seconds"] = prior_elapsed + time.perf_counter()-start_time
        atomic_save(path,{"signature":signature,"source_hash":source_hash(),"spec":spec,"config":cfg,
            "dataset_hash":train_store.manifest["dataset_hash"],"normalization":norm,
            "training_view_hash":train_store.view["hash"],
            "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "state":state,"rng":rng_state(),"sessions":sessions})

    pairs = len(train_store.rows)*train_store.steps
    try:
        while state["epoch"] < settings["epochs"]:
            model.train()
            # Deterministic epoch permutation + cursor makes mid-epoch restart exact on a fixed backend.
            permutation = np.random.default_rng(np.random.SeedSequence([spec["seed"],state["epoch"]])).permutation(pairs)
            for offset in range(state["cursor"],pairs,settings["batch_size"]):
                indices = permutation[offset:offset+settings["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0,len(indices),cfg["runtime"]["microbatch"]):
                    sub = indices[micro:micro+cfg["runtime"]["microbatch"]]
                    a,b = train_store.pairs(sub)
                    x,y = torch.as_tensor(a,device=device),torch.as_tensor(b,device=device)
                    total,supervised,pde = training_loss(model,x,y,norm,
                        cfg["data"]["dt"],cfg["data"]["viscosity"],spec)
                    if not torch.isfinite(total):
                        write_json(folder/"failure.json",{"reason":"Nonfinite training loss","epoch":state["epoch"],"offset":offset})
                        raise FloatingPointError(f"Nonfinite training loss in {spec['name']}")
                    (total*(len(sub)/len(indices))).backward()
                    state["epoch_sums"][0] += float(supervised)*len(sub)
                    state["epoch_sums"][1] += float(pde)*len(sub)
                    state["epoch_sums"][2] += len(sub)
                if not gradients_finite(model):
                    raise FloatingPointError("Nonfinite parameter gradient")
                optimizer.step()
                state["cursor"] = offset+len(indices)
                state["updates"] += 1
                if device.type == "cuda":
                    allocated = torch.cuda.max_memory_allocated(device)
                elif device.type == "mps":
                    allocated = torch.mps.current_allocated_memory()
                else:
                    allocated = 0
                state["sampled_device_memory_bytes"] = max(state["sampled_device_memory_bytes"],allocated)
                if state["updates"] % settings["checkpoint_every_updates"] == 0:
                    save(folder/"last.pt")
                    print(f"{spec['name']}: epoch {state['epoch']+1}/{settings['epochs']}, "
                          f"pairs {state['cursor']}/{pairs}, update {state['updates']} checkpointed",flush=True)
                if max_updates is not None and state["updates"] >= max_updates:
                    save(folder/"last.pt")
                    return {"status":"paused_for_test","updates":state["updates"]}
            used_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            state["epoch"] += 1
            state["cursor"] = 0
            sums = state["epoch_sums"]
            row = {"epoch":state["epoch"],"data_loss":sums[0]/sums[2],"physics_loss":sums[1]/sums[2],
                   "learning_rate":used_lr,"next_learning_rate":optimizer.param_groups[0]["lr"]}
            state["epoch_sums"] = [0.0,0.0,0]
            if state["epoch"] % settings["validation_every"] == 0 or state["epoch"] == settings["epochs"]:
                score = validation_score(model,val_store,cfg,device)
                row["validation_rollout_error"] = score
                if score < state["best_score"]:
                    state["best_score"] = score
                    state["best_epoch"] = state["epoch"]
                    state["history"].append(row)
                    save(folder/"best.pt")
                    state["history"].pop()
                print(f"{spec['name']}: epoch {state['epoch']}, validation={score:.6g}",flush=True)
            state["history"].append(row)
            save(folder/"last.pt")
            write_json(folder/"history.json",state["history"])
        result = {"status":"complete" if math.isfinite(state["best_score"]) else "failed_validation",
            "spec":spec,"signature":signature,"dataset_hash":train_store.manifest["dataset_hash"],
            "parameters":parameter_count(model),"best_score":state["best_score"],
            "best_epoch":state.get("best_epoch"),"epochs":state["epoch"],"updates":state["updates"],
            "elapsed_seconds":state["elapsed_seconds"],"sampled_device_memory_bytes":state["sampled_device_memory_bytes"],
            "host_process_peak_memory_bytes":host_peak_memory_bytes(),
            "device":str(device),"sessions":sessions}
        result.update(source_hash=source_hash(), training_view_hash=train_store.view["hash"],
                      best_checkpoint_hash=file_hash(folder/"best.pt") if (folder/"best.pt").exists() else None)
        write_json(folder/"result.json",result)
        return result
    finally:
        train_store.close(); val_store.close()


def make_scheduler(optimizer, settings):
    if settings.get("scheduler", "step") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"],
                                                        eta_min=settings.get("eta_min", 0.0))
    return torch.optim.lr_scheduler.StepLR(optimizer,settings["lr_step"],settings["lr_gamma"])


def lock_selection(cfg,root):
    root = Path(root)
    results = []
    for spec in matrix(cfg):
        path = root/"runs"/spec["name"]/"result.json"
        if not path.exists():
            raise ValueError(f"Complete the registered training matrix before opening test results: {spec['name']}")
        result = read_json(path)
        if result["status"] != "complete" or result["signature"] != experiment_signature(cfg):
            raise ValueError(f"Run incomplete or failed: {spec['name']}; record and resolve before locking selection")
        if result["spec"] != spec or result["epochs"] != cfg["training"]["epochs"]:
            raise ValueError(f"Run does not match registered protocol: {spec['name']}")
        if "variants" in cfg["training"]:
            if result.get("source_hash") != source_hash() or result.get("best_checkpoint_hash") != file_hash(root/"runs"/spec["name"]/"best.pt"):
                raise ValueError(f"Completed run source/checkpoint changed: {spec['name']}")
        results.append(result)
    if len({r["dataset_hash"] for r in results}) != 1 or len({r.get("training_view_hash") for r in results}) != 1:
        raise ValueError("Registered runs use different datasets/training subsets")
    selected = {}
    for method in ([] if "variants" in cfg["training"] else cfg["training"]["derivatives"]):
        scores = {}
        for weight in cfg["training"]["weights"]:
            values = [r["best_score"] for r in results if r["spec"]["weight"] == weight and r["spec"]["derivative"] == method]
            scores[str(weight)] = float(np.mean(values))
        selected[method] = {"weight":float(min(scores,key=lambda w:(scores[w],float(w)))),"validation_scores":scores}
    lock = {"signature":experiment_signature(cfg),"dataset_hash":results[0]["dataset_hash"],
            "selected":selected,"selection_rule":"mean validation rollout error across all seeds"}
    if "variants" in cfg["training"]:
        lock.update(selection_rule="predeclared variants; best validation checkpoint per run",
                    training_view_hash=results[0]["training_view_hash"],
                    checkpoints={r["spec"]["name"]:r["best_checkpoint_hash"] for r in results},
                    comparisons=cfg["evaluation"].get("comparisons", []))
    path = root/"selection.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("Selection changed after test access; use a new experiment version")
    write_json(path,lock)
    return lock
