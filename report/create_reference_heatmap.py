"""Plot the first stored test trajectory from each completed data source."""
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
studies = [
    ("Custom data", ROOT / "artifacts/eight-run/full/data/test-0000.h5", "test-00000", "ν = 0.01"),
    ("PDEBench", ROOT / "artifacts/pdebench/data/test-0000.h5", "test-09000", "ν = 0.001/π (configured)"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout="constrained",
                         gridspec_kw={"width_ratios": [1.45, 1]})
for row, (name, path, expected, viscosity) in enumerate(studies):
    with h5py.File(path, "r") as data:
        tid = data["ids"][0].decode()
        assert tid == expected
        u = data["u_256"][0].astype(float)
        x, t = data["x_256"][:], data["times"][:]
    assert u.shape == (len(t), len(x)) == (201, 256)
    # Include the actual grid origin: PDEBench uses cell-centered samples.
    dx = float(x[1] - x[0]); dt = float(t[1] - t[0])
    extent = [x[0]-dx/2, x[-1]+dx/2, t[0]-dt/2, t[-1]+dt/2]
    image = axes[row, 0].imshow(u, origin="lower", aspect="auto", extent=extent,
                                cmap="viridis", vmin=np.min(u), vmax=np.max(u),
                                interpolation="nearest")
    axes[row, 0].axhline(1, color="black", linestyle="--", linewidth=.8, alpha=.7)
    axes[row, 0].set(xlabel="Position x", ylabel="Time t", xlim=(0, 1), ylim=(0, 2),
                     title=f"{name}: {tid} | {viscosity} | N = 256")
    color = fig.colorbar(image, ax=axes[row, 0], pad=.02)
    color.set_label("Reference field u(x,t)")
    axes[row, 1].plot(x, u[0], color="#1e6e98", label="Initial, t = 0")
    axes[row, 1].plot(x, u[-1], color="#c35b32", label="Final, t = 2")
    axes[row, 1].set(xlabel="Position x", ylabel="Reference field u(x,t)", xlim=(0, 1),
                     title=f"{name}: initial and final profiles")
    axes[row, 1].legend(frameon=False, fontsize=9)
    axes[row, 1].spines[["top", "right"]].set_visible(False)
fig.suptitle("Example reference trajectories at the training-grid resolution", fontsize=15, weight="bold")
FIG.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"03_04_reference_solutions.{ext}", dpi=220, facecolor="white")
plt.close(fig)
print("Created reference trajectory figures for", ", ".join(s[2] for s in studies))
