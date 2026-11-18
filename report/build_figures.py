"""Build report figures/tables from saved results; never train or evaluate models.

Run from the project root: MPLCONFIGDIR=/tmp/burgers-report-mpl .venv/bin/python report/build_figures.py
Only report/ is written. Existing experiment artifacts are read without modification.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
EVIDENCE = OUT / "evidence"
EVIDENCE.mkdir(exist_ok=True)
STUDIES = {
    "custom2000": ("Custom 2,000 · two seeds", "eight-run/full", [0, 1]),
    "custom200": ("Custom 200 · seed 0", "custom-200trajectories-seed0", [0]),
    "pdebench": ("PDEBench 200 · two seeds", "pdebench", [0, 1]),
}
MODELS = ["unet", "fno", "physics-soft", "physics-constrained"]
LABELS = ["U-Net", "FNO", "Soft FNO", "Constrained FNO"]
COLORS = ["#2878a6", "#e48726", "#8b63aa", "#208574"]
MANIFEST = {}


def read(path, expected=None):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    MANIFEST[str(path.relative_to(ROOT))] = sha
    return json.loads(data)


SUMMARIES = {key: read(ROOT / "artifacts" / folder / "summary/summary.json")
             for key, (_, folder, _) in STUDIES.items()}
assert all(not s["fixture"] for s in SUMMARIES.values())


def row(study, model, metric="rollout_error", horizon=100, n=256, adapter="native"):
    rows = [r for r in SUMMARIES[study]["table"] if
            (r["family"], r["metric"], r["horizon"], r["resolution"], r["adapter"])
            == (model, metric, horizon, n, adapter)]
    assert len(rows) == 1
    return rows[0]


def source(study, relative):
    path = ROOT / "artifacts" / STUDIES[study][1] / relative
    expected = SUMMARIES[study]["provenance"].get(relative)
    return read(path, expected)


def number(value, precision=6):
    return "Unavailable: failed rollout(s)" if value is None else f"{value:.{precision}g}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


TABLES = {}
TABLES["headline"] = table(["Model", "Custom 2,000 (2 seeds)", "Custom 200 (seed 0)", "PDEBench 200 (2 seeds)"],
    [[label] + [number(row(s, m)["mean"], 8) for s in STUDIES]
     for m, label in zip(MODELS, LABELS)])
for study in STUDIES:
    TABLES[study] = table(["Model", "100-step mean", "100-step seed SD", "200-step mean", "t > 1 mean", "200-step failures"],
        [[label, number(row(study, m)["mean"], 8),
          "Not estimable (one seed)" if len(STUDIES[study][2]) == 1 else number(row(study, m)["std"]),
          number(row(study, m, horizon=200)["mean"], 8),
          number(row(study, m, "extrapolation_error", 200)["mean"], 8),
          f"{100 * row(study, m, horizon=200)['failure_rate']:.2f}%"]
         for m, label in zip(MODELS, LABELS)])
TABLES["lowdata"] = table(["Model", "2,000 trajectories, seed 0", "200 trajectories, seed 0", "Error multiplier"],
    [[label, number(row("custom2000", m)["seed_means"][0], 8),
      number(row("custom200", m)["mean"], 8),
      f"{row('custom200', m)['mean'] / row('custom2000', m)['seed_means'][0]:.2f}×"]
     for m, label in zip(MODELS, LABELS)])
TABLES["physics"] = table(["Study", "Model", "Normalized maximum mass drift", "Energy error", "Spectral residual RMS (time mean)"],
    [[STUDIES[s][0], label] + [number(row(s, m, metric)["mean"])
      for metric in ["normalized_mass_drift", "energy_error", "spectral_residual"]]
     for s in STUDIES for m, label in zip(MODELS, LABELS)])
TABLES["effects"] = table(["Study", "Comparison", "100-step error difference", "Paired bootstrap 95% interval"],
    [[STUDIES[s][0], r["variant"] + " minus " + r["baseline"], number(r["difference"]),
      "[" + ", ".join(number(v) for v in r["ci95"]) + "]"]
     for s in STUDIES for r in SUMMARIES[s]["comparisons"]
     if r["resolution"] == 256 and r["horizon"] == 100 and r["metric"] == "rollout_error"])
TABLES["resolution"] = table(["Study", "Model", "Execution", "N=256", "N=512", "N=1024"],
    [[STUDIES[s][0], label, adapter] +
     [number(row(s, m, n=n, adapter="native" if n == 256 else adapter)["mean"])
      for n in [256, 512, 1024]]
     for s in ["custom2000", "pdebench"] for m, label in zip(MODELS, LABELS)
     for adapter in (["native", "cubic"] if m == "unet" else ["native"])])

COSTS = {}
for s, (_, folder, seeds) in STUDIES.items():
    for m in MODELS:
        COSTS[s, m] = [read(ROOT / "artifacts" / folder / "runs" / f"{m}-s{seed}" / "result.json")
                       for seed in seeds]
        assert all(r["status"] == "complete" and r["epochs"] == 60 for r in COSTS[s, m])
TABLES["cost"] = table(["Model", "Parameters (real degrees of freedom)", "Custom 2,000 minutes", "Custom 200 minutes", "PDEBench 200 minutes"],
    [[label, COSTS["custom2000", m][0]["parameters"]] +
     [f"{np.mean([r['elapsed_seconds'] for r in COSTS[s, m]]) / 60:.2f}" for s in STUDIES]
     for m, label in zip(MODELS, LABELS)])

# Cross-study bars: error bars describe seed SD, not confidence intervals.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
for ax, s in zip(axes, STUDIES):
    vals = [row(s, m)["mean"] for m in MODELS]
    sd = [row(s, m)["std"] or 0 for m in MODELS]
    ax.bar(range(4), vals, color=COLORS,
           yerr=sd if len(STUDIES[s][2]) > 1 else None, capsize=4)
    ax.set(xticks=range(4), xticklabels=["U-Net", "FNO", "Soft\nFNO", "Constrained\nFNO"],
           ylabel="100-step mean relative L2 error", title=STUDIES[s][0])
    ax.set_ylim(0, max(np.array(vals) + sd) * 1.25)
    for i, v in enumerate(vals):
        ax.text(i, v + sd[i] + max(vals) * .035, f"{v:.4g}", ha="center", fontsize=8)
save(fig, "04_01_accuracy_by_study")

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for m, label, color in zip(MODELS, LABELS, COLORS):
    ax.plot([200, 2000], [row("custom200", m)["mean"], row("custom2000", m)["seed_means"][0]],
            "o-", color=color, label=label)
ax.set(xscale="log", yscale="log", xticks=[200, 2000], xticklabels=["200", "2,000"],
       xlabel="Training trajectories (seed 0, 60 epochs)", ylabel="100-step mean relative L2 error",
       title="Reduced-data comparison: data and update budgets both change")
ax.legend(fontsize=9)
save(fig, "04_02_reduced_data")

# Read native N=256 curves, and verify each source against its saved summary hash.
CURVES, FAILED = {}, []
for s, (_, _, seeds) in STUDIES.items():
    for m in MODELS:
        seed_values = []
        for seed in seeds:
            doc = source(s, f"evaluation/{m}-s{seed}/n256-native.json")
            assert len(doc["records"]) == 500
            seed_values.append(np.asarray([r["curves"]["relative_l2"] for r in doc["records"]], dtype=float))
            for record in doc["records"]:
                if record["horizons"]["200"]["failed"]:
                    FAILED.append({"study": s, "model": m, "seed": seed, "record": record})
        CURVES[s, m] = np.asarray(seed_values)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
for ax, s in zip(axes, STUDIES):
    for m, label, color in zip(MODELS, LABELS, COLORS):
        values = CURVES[s, m].mean(axis=(0, 1))  # Deliberately no nanmean / survivor-only averaging.
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad):
            values[bad[0]:] = np.nan
        ax.plot(np.arange(1, len(values)) * .01, values[1:], label=label, color=color)
    ax.axvline(1, color="0.4", ls="--", lw=1)
    ax.set(yscale="log", xlabel="Time", ylabel="Mean relative L2 error", title=STUDIES[s][0])
axes[0].legend(fontsize=8)
axes[2].text(.03, .97, "Constrained mean ends at first failure;\nfailed trajectories are not removed.",
             transform=axes[2].transAxes, va="top", fontsize=8)
save(fig, "04_03_rollout_error")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
for ax, s in zip(axes, STUDIES):
    ax.bar(range(4), [row(s, m, "normalized_mass_drift")["mean"] for m in MODELS], color=COLORS)
    ax.set(yscale="log", xticks=range(4), xticklabels=["U-Net", "FNO", "Soft\nFNO", "Constrained\nFNO"],
           ylabel="Mean normalized maximum mass drift", title=STUDIES[s][0])
save(fig, "04_04_mass_conservation")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
for ax, s in zip(axes, STUDIES):
    ax.bar(range(4), [row(s, m, "energy_error")["mean"] for m in MODELS], color=COLORS)
    ax.set(yscale="log", xticks=range(4), xticklabels=["U-Net", "FNO", "Soft\nFNO", "Constrained\nFNO"],
           ylabel="Mean absolute energy error", title=STUDIES[s][0])
save(fig, "04_05_energy_error")

CONTROLS = {}
for s in ["custom2000", "pdebench"]:
    for n in [256, 512, 1024]:
        for control in ["reference", "oracle_cubic"]:
            doc = source(s, f"evaluation/controls/n{n}-{control}.json")
            CONTROLS[s, n, control] = {
                metric: float(np.mean([r["horizons"]["100"][metric] for r in doc["records"]]))
                for metric in ["rollout_error", "spectral_residual", "finite_difference_residual"]}
TABLES["controls"] = table(["Study", "N", "Reference spectral residual", "Reference FD residual", "Oracle cubic error"],
    [[STUDIES[s][0], n, number(CONTROLS[s, n, "reference"]["spectral_residual"]),
      number(CONTROLS[s, n, "reference"]["finite_difference_residual"]),
      number(CONTROLS[s, n, "oracle_cubic"]["rollout_error"])]
     for s in ["custom2000", "pdebench"] for n in [256, 512, 1024]])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
for ax, s in zip(axes, ["custom2000", "pdebench"]):
    for m, label, color in zip(MODELS, LABELS, COLORS):
        ax.scatter(row(s, m, "spectral_residual")["mean"], row(s, m)["mean"], color=color, s=65, label=label)
    ax.axvline(CONTROLS[s, 256, "reference"]["spectral_residual"], color="0.35", ls="--", label="Reference residual")
    ax.set(xscale="log", yscale="log", xlabel="Mean spectral residual RMS", ylabel="100-step relative L2 error",
           title=STUDIES[s][0])
axes[0].legend(fontsize=8)
save(fig, "04_06_residual_accuracy")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
for ax, s in zip(axes, ["custom2000", "pdebench"]):
    for m, label, color in zip(MODELS, LABELS, COLORS):
        for adapter in (["native", "cubic"] if m == "unet" else ["native"]):
            values = [row(s, m, n=n, adapter="native" if n == 256 else adapter)["mean"] for n in [256, 512, 1024]]
            ax.plot([256, 512, 1024], values, "o--" if adapter == "cubic" else "o-",
                    color=color, label=f"{label} ({adapter})")
    ax.set(yscale="log", xticks=[256, 512, 1024], xlabel="Evaluation grid points", ylabel="100-step relative L2 error",
           title=STUDIES[s][0])
axes[0].legend(fontsize=8)
save(fig, "04_07_resolution_transfer")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
for ax, case in zip(axes, FAILED):
    r = case["record"]
    values = np.asarray(r["curves"]["relative_l2"], dtype=float)
    first = int(np.flatnonzero(~np.isfinite(values))[0])
    ax.plot(np.arange(first) * .01, values[:first], color=COLORS[3])
    ax.axvline(first * .01, color="#b52a3e", ls="--", label=f"Failure at step {first}")
    ax.axvline(1, color="0.5", ls=":", label="Training horizon")
    ax.set(yscale="log", xlabel="Time", ylabel="Trajectory relative L2 error",
           title=f"{r['id']} · constrained FNO seed {case['seed']}")
    ax.legend(fontsize=8)
assert len(FAILED) == 2, "Review failure figure layout if source failures change."
save(fig, "04_08_failure_cases")
TABLES["failures"] = table(["Test trajectory", "Seed", "First failed step", "First failed time", "100-step mean error"],
    [[c["record"]["id"], c["seed"],
      int(np.flatnonzero(~np.isfinite(np.asarray(c["record"]["curves"]["relative_l2"], dtype=float)))[0]),
      f"{int(np.flatnonzero(~np.isfinite(np.asarray(c['record']['curves']['relative_l2'], dtype=float)))[0]) * .01:.2f}",
      number(c["record"]["horizons"]["100"]["rollout_error"])] for c in FAILED])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
for ax, s in zip(axes, STUDIES):
    for m, label, color in zip(MODELS, LABELS, COLORS):
        minutes = np.mean([r["elapsed_seconds"] for r in COSTS[s, m]]) / 60
        ax.scatter(minutes, row(s, m)["mean"], color=color, s=65, label=label)
    ax.set(yscale="log", xlabel="Recorded training + validation minutes", ylabel="100-step relative L2 error",
           title=STUDIES[s][0])
axes[0].legend(fontsize=8)
save(fig, "04_09_accuracy_cost")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
for ax, (s, (_, folder, seeds)) in zip(axes, STUDIES.items()):
    for m, label, color in zip(MODELS, LABELS, COLORS):
        for seed in seeds:
            history = read(ROOT / "artifacts" / folder / "runs" / f"{m}-s{seed}" / "history.json")
            h = [r for r in history if "validation_rollout_error" in r]
            ax.plot([r["epoch"] for r in h], [r["validation_rollout_error"] for r in h],
                    color=color, ls="-" if seed == 0 else "--", label=label if seed == 0 else None, lw=1)
    ax.set(yscale="log", xlabel="Epoch", ylabel="Validation rollout error", title=STUDIES[s][0])
axes[0].legend(fontsize=8)
save(fig, "03_03_validation_history")

# Export the exact plotted/tabulated source rows for inspection.
with (EVIDENCE / "metrics.csv").open("w", newline="") as f:
    fields = ["study"] + list(next(iter(SUMMARIES.values()))["table"][0])
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for s in STUDIES:
        for r in SUMMARIES[s]["table"]:
            writer.writerow({"study": s, **r})
(EVIDENCE / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in TABLES.items()) + "\n")
(EVIDENCE / "source_manifest.json").write_text(json.dumps({
    "description": "SHA-256 of saved experimental inputs consumed by build_figures.py; no model execution",
    "sources": MANIFEST,
    "figures": sorted(p.name for p in FIG.glob("*.png")),
}, indent=2) + "\n")
report = OUT / "FINAL_REPORT.md"
if report.exists():
    content = report.read_text()
    for name, value in TABLES.items():
        pattern = rf"(<!-- BEGIN GENERATED {name} -->).*?(<!-- END GENERATED {name} -->)"
        content, count = re.subn(pattern, lambda m: m[1] + "\n\n" + value + "\n\n" + m[2], content, flags=re.S)
        assert count == 1, f"Expected exactly one report table block: {name}"
    report.write_text(content)
print(f"Created {len(list(FIG.glob('*.png')))} figures (PNG + PDF), {len(TABLES)} tables, and evidence exports.")
