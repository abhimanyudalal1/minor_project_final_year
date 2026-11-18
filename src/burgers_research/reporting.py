from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import digest, file_hash, matrix, read_json, source_hash, write_json
from .training import lock_selection


def bootstrap_difference(a,b,samples,seed):
    """Paired resampling of seeds and shared trajectories; no timestep resampling."""
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Expected aligned [seed,trajectory] arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"difference":None,"ci95":None,"reason":"failed/nonfinite rollouts; see failure rates"}
    delta = a-b
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for i in range(samples):
        seeds = rng.integers(delta.shape[0],size=delta.shape[0])
        trajectories = rng.integers(delta.shape[1],size=delta.shape[1])
        estimates[i] = delta[seeds[:,None],trajectories[None,:]].mean()
    return {"difference":float(delta.mean()),"ci95":np.quantile(estimates,[0.025,0.975]).tolist()}


def family(spec):
    if "family" in spec:
        return spec["family"]
    if spec["weight"] == 0:
        return spec["model"]
    return f"physics-{spec['derivative']}-l{spec['weight']:g}"


def summarize(cfg,root,analysis_root=None):
    root = Path(root)
    artifacts = Path(analysis_root or root)
    selection = lock_selection(cfg,root)
    context = read_json(artifacts/"evaluation-context.json")
    if context["hash"] != digest({k:v for k,v in context.items() if k != "hash"}) or context["source_hash"] != source_hash():
        raise ValueError("Evaluation context/source changed")
    if context["training_signature"] != selection["signature"] or context["training_dataset_hash"] != selection["dataset_hash"]:
        raise ValueError("Evaluation context does not match training selection")
    resolutions = context["config"]["data"]["resolutions"]
    groups = {}
    provenance = {}
    for spec in matrix(cfg):
        for n in resolutions:
            adapters = ["native"] if n == cfg["data"]["base_n"] else ["native","cubic","linear"]
            for adapter in adapters:
                path = artifacts/"evaluation"/spec["name"]/f"n{n}-{adapter}.json"
                doc = read_json(path)
                provenance[str(path.relative_to(artifacts))] = file_hash(path)
                if doc["dataset_hash"] != context["dataset_hash"] or doc.get("context_hash") != context["hash"] or doc["checkpoint_hash"] != file_hash(root/"runs"/spec["name"]/"best.pt"):
                    raise ValueError("Evaluation provenance mismatch")
                key = (family(spec),n,adapter)
                groups.setdefault(key,[]).append(doc)
    table = []
    for (name,n,adapter),docs in groups.items():
        docs.sort(key=lambda d:d["spec"]["seed"])
        for h in cfg["evaluation"]["horizons"]:
            metrics = ["rollout_error","terminal_error","normalized_mass_drift","energy_error",
                       "dissipation_balance_error","spectral_residual","finite_difference_residual"]
            if h > round(context["config"]["data"]["train_time"]/context["config"]["data"]["dt"]):
                metrics.append("extrapolation_error")
            for metric in metrics:
                values = np.array([[r["horizons"][str(h)][metric] if r["horizons"][str(h)][metric] is not None else np.inf
                                    for r in doc["records"]] for doc in docs])
                failures = np.array([[r["horizons"][str(h)]["failed"] for r in doc["records"]] for doc in docs])
                seed_means = values.mean(1)
                valid = np.isfinite(seed_means).all() and not failures.any()
                table.append({"family":name,"resolution":n,"adapter":adapter,"horizon":h,"metric":metric,
                    "mean":float(seed_means.mean()) if valid else None,
                    "std":float(seed_means.std(ddof=1)) if valid and len(docs)>1 else None,
                    "failure_rate":float(failures.mean()),"seed_means":seed_means.tolist()})
    comparisons = []
    pairs = (cfg["evaluation"].get("comparisons", []) if "variants" in cfg["training"] else
             [(f"physics-{method}-l{choice['weight']:g}", "fno") for method,choice in selection["selected"].items()])
    for chosen, baseline in pairs:
        for n in resolutions:
            base = groups[(baseline,n,"native")]
            variant = groups[(chosen,n,"native")]
            for x,y in zip(base,variant):
                assert x["spec"]["seed"] == y["spec"]["seed"]
                assert [r["id"] for r in x["records"]] == [r["id"] for r in y["records"]]
            for h in cfg["evaluation"]["horizons"]:
                metrics = ["rollout_error","normalized_mass_drift","energy_error"]
                if h > round(context["config"]["data"]["train_time"]/context["config"]["data"]["dt"]):
                    metrics.append("extrapolation_error")
                for metric in metrics:
                    def values(docs):
                        return [[r["horizons"][str(h)][metric] if not r["horizons"][str(h)]["failed"]
                                 and r["horizons"][str(h)][metric] is not None else np.inf
                                 for r in doc["records"]] for doc in docs]
                    comparisons.append({"variant":chosen,"baseline":baseline,"resolution":n,"horizon":h,"metric":metric,
                        **bootstrap_difference(values(variant),values(base),cfg["evaluation"]["bootstrap_samples"],cfg["evaluation"]["bootstrap_seed"])})
    for n in resolutions:
        for control in ("reference","persistence","oracle_cubic","oracle_linear"):
            path = artifacts/"evaluation"/"controls"/f"n{n}-{control}.json"
            doc = read_json(path)
            if doc["dataset_hash"] != context["dataset_hash"] or doc.get("context_hash") != context["hash"]:
                raise ValueError("Control dataset mismatch")
            provenance[str(path.relative_to(artifacts))] = file_hash(path)
    provenance["evaluation-context.json"] = file_hash(artifacts/"evaluation-context.json")
    result = {"fixture":cfg["fixture"],"selection":selection,"table":table,"comparisons":comparisons,
              "provenance":provenance,"evaluation_context":context}
    write_json(artifacts/"summary"/"summary.json",result)
    with open(artifacts/"summary"/"metrics.csv","w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    return result


def build_report(cfg,root,analysis_root=None):
    # Agg is set before pyplot import so reporting works on headless Colab/Kaggle.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    artifacts = Path(analysis_root or root)
    summary = read_json(artifacts/"summary"/"summary.json")
    context = summary["evaluation_context"]
    resolutions = context["config"]["data"]["resolutions"]
    for filename,sha in summary["provenance"].items():
        if file_hash(artifacts/filename) != sha:
            raise ValueError("Artifacts changed after summarization")
    folder = artifacts/"report"
    folder.mkdir(parents=True,exist_ok=True)
    selection = summary["selection"]
    chosen = {f"physics-{method}-l{row['weight']:g}" for method,row in selection["selected"].items()}
    if "variants" in cfg["training"]:
        chosen = {v["name"] for v in cfg["training"]["variants"]}
    main = [s for s in matrix(cfg) if family(s) in chosen or s["weight"] == 0]
    architecture = {family(s): s["model"] for s in main}
    curves = {"relative_l2":"Rollout relative L2 error","mass_drift":"Signed mass drift", "energy":"Kinetic energy",
              "dissipation_balance":"Integrated dissipation balance error"}
    base = cfg["data"]["base_n"]
    for metric,label in curves.items():
        fig,ax = plt.subplots(figsize=(7,4))
        for name in sorted({family(s) for s in main}):
            seed_curves = []
            for spec in main:
                if family(spec) != name:
                    continue
                doc = read_json(artifacts/"evaluation"/spec["name"]/f"n{base}-native.json")
                values = np.array([[np.nan if v is None else v for v in r["curves"][metric]] for r in doc["records"]])
                seed_curves.append(np.mean(values,axis=0))  # NaNs remain visible; no survivor-only mean.
            values = np.array(seed_curves)
            t = np.arange(values.shape[-1])*cfg["data"]["dt"]
            avg,std = values.mean(0),values.std(0,ddof=1)
            ax.plot(t,avg,label=name)
            ax.fill_between(t,avg-std,avg+std,alpha=0.15)
        if metric == "energy":
            control = read_json(artifacts/"evaluation"/"controls"/f"n{base}-reference.json")
            reference = np.mean([r["curves"]["energy"] for r in control["records"]],axis=0)
            ax.plot(np.arange(len(reference))*cfg["data"]["dt"],reference,"k--",label="Numerical reference")
        ax.set(xlabel="Time",ylabel=label,title="Integration fixture — not research evidence" if cfg["fixture"] else label)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/f"{metric}.png",dpi=160); plt.close(fig)
    for plot,metric,xkey in (("resolution","rollout_error","resolution"),("lambda","rollout_error","weight")):
        if plot == "lambda" and "variants" in cfg["training"]:
            continue
        fig,ax = plt.subplots(figsize=(7,4))
        h = cfg["evaluation"]["horizons"][0]
        if plot == "resolution":
            for name in sorted({family(s) for s in main}):
                rows = [r for r in summary["table"] if r["family"] == name and r["metric"] == metric and r["horizon"] == h
                        and r["adapter"] == ("cubic" if architecture[name] == "unet" and r["resolution"] != base else "native")]
                ax.plot([r["resolution"] for r in rows],[np.nan if r["mean"] is None else r["mean"] for r in rows],"o-",label=name)
        else:
            for method in cfg["training"]["derivatives"]:
                weights = cfg["training"]["weights"]
                scores = selection["selected"][method]["validation_scores"]
                ax.plot(weights,[scores[str(w)] for w in weights],"o-",label=method)
            ax.set_xscale("log")
        ax.set(xlabel=xkey,ylabel="Validation rollout error" if plot == "lambda" else "Test rollout error")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/f"{plot}.png",dpi=160); plt.close(fig)
    for name,xmetric in (("residual_vs_drift","spectral_residual"),("accuracy_vs_cost",None)):
        fig,ax = plt.subplots(figsize=(7,4))
        for spec in main:
            result = read_json(root/"runs"/spec["name"]/"result.json")
            doc = read_json(artifacts/"evaluation"/spec["name"]/f"n{base}-native.json")
            records = [r["horizons"][str(cfg["evaluation"]["horizons"][0])] for r in doc["records"]]
            if any(r["failed"] for r in records):
                continue
            x = np.mean([r[xmetric] for r in records]) if xmetric else result["elapsed_seconds"]
            y = np.mean([r["normalized_mass_drift" if xmetric else "rollout_error"] for r in records])
            ax.scatter(x,y,label=spec["name"],s=20)
        ax.set(xlabel="Spectral residual RMS" if xmetric else "Training + validation elapsed seconds",
               ylabel="Normalized mass drift" if xmetric else "Rollout error")
        ax.legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(folder/f"{name}.png",dpi=160); plt.close(fig)
    def fmt(x):
        return "FAILED / unavailable" if x is None else f"{x:.6g}"
    lines = ["# Burgers neural-operator research report", "",
        "**INTEGRATION FIXTURE: these results are not scientific evidence.**" if cfg["fixture"] else "**Production experiment results.**", "",
        "## Research questions", "",
        "Does a soft conservative PDE penalty improve autoregressive accuracy, conservation, stability, and grid transfer? "
        "How sensitive are the outcomes to residual discretization and penalty strength?", "",
        "## Methods and provenance", "",
        f"Protocol: `{cfg['protocol']}`. Training dataset: `{selection['dataset_hash']}`. "
        f"Evaluation dataset: `{context['dataset_hash']}`. Shift evaluation: {context['shifted']}. "
        f"All {len(list(matrix(cfg)))} registered runs completed before test selection was opened.", "",
        "Methods, architecture adaptations, numerical verification, and limitations are documented in the project's docs directory. "
        "This report is generated from saved artifacts; it does not infer statistical significance from point estimates.", "",
        f"The training equation is u_t + (u²/2)_x = {cfg['data']['viscosity']} u_xx on [0,1). "
        f"Evaluation viscosity is {context['config']['data']['viscosity']}; "
        f"evaluation test trajectories: {context['config']['data']['counts']['test']}. "
        f"Trajectory counts: {cfg['data']['counts']}; reference N={cfg['data']['reference_n']}; "
        f"model grids={cfg['data']['resolutions']}; output dt={cfg['data']['dt']}. "
        f"Training uses {cfg['training'].get('train_trajectories',cfg['data']['counts']['train'])} trajectories, "
        f"t≤{cfg['data']['train_time']}, {cfg['training']['epochs']} epochs, "
        f"effective batch {cfg['training']['batch_size']}, and seeds {cfg['training']['seeds']}.", "",
        "## Validation selection", "",
        *[f"- {method}: lambda={row['weight']:g}, selected using all seeds." for method,row in selection["selected"].items()], "",
        "## Main results", "", "U-Net variants use cubic adaptation above the training grid; FNO variants use native execution. "
        "The paired-effect tables below compare native execution for both models; all adapters are retained in the full summary.", "",
        "| Model | N | Adapter | Steps | Rollout mean | Seed SD | Failure rate |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary["table"]:
        primary_adapter = "cubic" if architecture.get(row["family"]) == "unet" and row["resolution"] != base else "native"
        if row["family"] in chosen|{"fno","unet"} and row["metric"] == "rollout_error" and row["adapter"] == primary_adapter:
            lines.append(f"| {row['family']} | {row['resolution']} | {row['adapter']} | {row['horizon']} | {fmt(row['mean'])} | {fmt(row['std'])} | {row['failure_rate']:.3%} |")
    lines += ["", "## Paired effects", "", "Differences are the named variant minus its named baseline; negative values favor the variant. "
        f"Intervals resample seeds and trajectories as paired units. Only {len(cfg['training']['seeds'])} seeds limit uncertainty estimation.", "",
              "| Variant | Baseline | N | Steps | Metric | Difference | 95% interval |", "|---|---|---:|---:|---|---:|---|"]
    for row in summary["comparisons"]:
        lines.append(f"| {row['variant']} | {row['baseline']} | {row['resolution']} | {row['horizon']} | {row['metric']} | {fmt(row['difference'])} | {row['ci95']} |")
    lines += ["", "## Out-of-training-horizon accuracy", "",
              "This averages only steps after the training time window, while continuing the rollout from t=0 without resetting.", "",
              "| Model | N | Adapter | Steps | Extrapolation error | Failure rate |", "|---|---:|---|---:|---:|---:|"]
    for row in summary["table"]:
        if row["metric"] == "extrapolation_error":
            lines.append(f"| {row['family']} | {row['resolution']} | {row['adapter']} | {row['horizon']} | {fmt(row['mean'])} | {row['failure_rate']:.3%} |")
    lines += ["", "## Reference and interpolation controls", "",
              "| Control | N | Steps | Rollout error | Spectral residual | FD residual |",
              "|---|---:|---:|---:|---:|---:|"]
    for n in resolutions:
        for control in ("reference","persistence","oracle_cubic","oracle_linear"):
            doc = read_json(artifacts/"evaluation"/"controls"/f"n{n}-{control}.json")
            for h in cfg["evaluation"]["horizons"]:
                rows = [r["horizons"][str(h)] for r in doc["records"]]
                averages = [np.mean([r[k] for r in rows]) for k in ("rollout_error","spectral_residual","finite_difference_residual")]
                lines.append(f"| {control} | {n} | {h} | {fmt(averages[0])} | {fmt(averages[1])} | {fmt(averages[2])} |")
    lines += ["", "## Architecture and compute", "",
              "Memory values are sampled live device allocations (CUDA uses allocator peak), "
              "and host peak RSS covers the entire process. Timings include validation and checkpoint overhead.", "",
              "| Run | Real parameters | Backend | Train + validation seconds | Sampled device bytes |",
              "|---|---:|---|---:|---:|"]
    for spec in main:
        result = read_json(root/"runs"/spec["name"]/"result.json")
        lines.append(f"| {spec['name']} | {result['parameters']} | {result['device']} | {result['elapsed_seconds']:.3f} | {result['sampled_device_memory_bytes']} |")
    lines += ["", "## Figures", ""]
    for name in list(curves)+["resolution","lambda","residual_vs_drift","accuracy_vs_cost"]:
        if name == "lambda" and "variants" in cfg["training"]:
            continue
        lines += [f"![{name}]({name}.png)", ""]
    lines += ["## Interpretation and limitations", "",
        "Claims apply only to the recorded training/evaluation distributions and tested horizons. "
        "The full sensitivity table, per-seed metrics, failure flags, control results, and reference residual floors must accompany interpretation. "
        "Mass drift and residual are related through the zero Fourier mode; a small soft penalty is not exact conservation. "
        "For a mean-projected variant, conservation is enforced up to floating-point roundoff and is not evidence of learned accuracy. "
        "The constrained variant combines changes; the core comparison cannot identify each change's individual effect. "
        "Reference projection and time-discretization error are distinct from learned error. "
        "The PINO-style objective is an adaptation, not a reproduction of the full PINO method. "
        "Timing includes the documented execution backend; cross-device timings are not a controlled speed comparison.", "",
        "## References", "",
        "- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895)\n"
        "- [U-Net](https://arxiv.org/abs/1505.04597)\n"
        "- [Physics-Informed Neural Operator](https://arxiv.org/abs/2111.03794)\n"
        "- [ETDRK4](https://doi.org/10.1137/S1064827502410633)\n"
        "- [Exactly conservative networks](https://arxiv.org/abs/2311.14131)", "",
        "## Artifact index", "", "See `../summary/metrics.csv`, `../summary/summary.json`, "
        "`../verification/solver.json`, `../data/manifest.json`, and `../runs/` for the complete audit trail.", ""]
    (folder/"report.md").write_text("\n".join(lines))
    write_json(folder/"manifest.json",{"summary_sha256":file_hash(artifacts/"summary"/"summary.json"),
        "files":{p.name:file_hash(p) for p in sorted(folder.iterdir()) if p.name != "manifest.json"}})
    return folder/"report.md"
