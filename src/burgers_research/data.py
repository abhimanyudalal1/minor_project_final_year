from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import h5py
import numpy as np

from .common import data_signature, digest, environment, file_hash, read_json, write_json
from .numerics import derivative, initial_condition, physical_checks, project, solve
from .storage import check_storage


def split_manifest(cfg):
    rng = np.random.default_rng(cfg["data"]["seed"])
    rows = []
    used = set(cfg["solver"]["verification_seeds"])
    for split, count in cfg["data"]["counts"].items():
        for i in range(count):
            seed = int(rng.integers(0, 2**63-1))
            while seed in used:
                seed = int(rng.integers(0, 2**63-1))
            used.add(seed)
            rows.append({"id": f"{split}-{i:05d}", "split": split, "seed": seed})
    return rows


def generate(cfg, root):
    root = Path(root)
    d = cfg["data"]
    verification = read_json(root/"verification"/"solver.json")
    if not verification["passed"] or verification["data_signature"] != data_signature(cfg):
        raise ValueError("A passing solver verification for this configuration is required.")
    folder = root/"data"
    if folder.is_symlink():
        raise ValueError("Cannot generate into a linked dataset; use reuse-data and validate-data")
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder/"manifest.json").exists():
        check_storage(cfg, root)
    splits = split_manifest(cfg)
    split_doc = {"data_signature": data_signature(cfg), "trajectories": splits}
    split_path = folder/"splits.json"
    if split_path.exists() and read_json(split_path) != split_doc:
        raise ValueError("Work directory belongs to a different dataset; select a new work directory.")
    write_json(split_path, split_doc)
    times = np.arange(round(d["final_time"]/d["dt"])+1) * d["dt"]
    shards = []
    for split in d["counts"]:
        rows = [r for r in splits if r["split"] == split]
        for start in range(0, len(rows), d["shard_size"]):
            subset = rows[start:start+d["shard_size"]]
            filename = f"{split}-{start//d['shard_size']:04d}.h5"
            path = folder/filename
            sidecar = path.with_suffix(".json")
            if path.exists() and sidecar.exists():
                info = read_json(sidecar)
                if info["sha256"] != file_hash(path) or info["data_signature"] != data_signature(cfg):
                    raise ValueError(f"Corrupt or incompatible shard: {path}")
                shards.append(info)
                continue
            before = time.perf_counter()
            checks = []
            shock_checks = []
            projection_errors = []
            temporary = path.with_suffix(".partial")
            with h5py.File(temporary, "w") as f:
                f.attrs["data_signature"] = data_signature(cfg)
                f.attrs["viscosity"] = d["viscosity"]
                f.attrs["internal_dt"] = verification["internal_dt"]
                f.create_dataset("times", data=times, track_times=False)
                f.create_dataset("ids", data=np.array([r["id"] for r in subset], dtype="S40"), track_times=False)
                f.create_dataset("seeds", data=[r["seed"] for r in subset], track_times=False)
                reference = f.create_dataset("reference", (len(subset),len(times),d["reference_n"]),
                    dtype="f8", chunks=(1,1,d["reference_n"]), compression="lzf", track_times=False)
                f.create_dataset("x_reference", data=np.arange(d["reference_n"])/d["reference_n"], track_times=False)
                grids = {}
                for n in d["resolutions"]:
                    grids[n] = f.create_dataset(f"u_{n}", (len(subset),len(times),n),
                        dtype="f4", chunks=(1,min(16,len(times)),n), compression="lzf", track_times=False)
                    f.create_dataset(f"x_{n}", data=np.arange(n)/n, track_times=False)
                for i, row in enumerate(subset):
                    u = solve(initial_condition(d["reference_n"], row["seed"], d), d["viscosity"],
                              d["dt"], d["final_time"], verification["internal_dt"])
                    check = physical_checks(u)
                    if check["mass_drift"] > cfg["solver"]["mass_tolerance"] or check["energy_increase"] > cfg["solver"]["energy_tolerance"]:
                        raise ValueError(f"Physical checks failed for {row['id']}; retain the seed and refine the solver.")
                    checks.append(check)
                    if cfg.get("evaluation_only", False):
                        spectrum = np.fft.rfft(u, axis=-1, norm="forward")
                        energy = 2*np.abs(spectrum)**2
                        energy[:,0] = 0  # Fluctuating energy, excluding the conserved mean.
                        energy[:,-1] *= 0.5
                        cutoff = cfg["model"]["fno"]["modes"]
                        fraction = energy[:,cutoff:].sum(-1)/np.maximum(energy.sum(-1),1e-12*max(energy[0].sum(),1e-24))
                        shock_checks.append({"id":row["id"],"cutoff_mode":cutoff,
                            "peak_high_mode_energy_fraction":float(fraction.max()),
                            "peak_abs_gradient":float(np.max(np.abs(derivative(u))))})
                    reference[i] = u
                    errors = {}
                    for n, target in grids.items():
                        low = project(u, n)
                        target[i] = low.astype(np.float32)
                        errors[str(n)] = float(np.sqrt(np.mean((u-project(low,d["reference_n"]))**2)))
                    projection_errors.append(errors)
            temporary.replace(path)
            info = {"file": filename, "sha256": file_hash(path), "data_signature": data_signature(cfg),
                    "split": split, "ids": [r["id"] for r in subset], "checks": checks,
                    "projection_rmse": projection_errors, "generation_seconds": time.perf_counter()-before}
            if shock_checks:
                info["shock_diagnostics"] = shock_checks
            write_json(sidecar, info)
            shards.append(info)
            print(f"Generated {filename} ({len(subset)} trajectories)", flush=True)
    train_steps = round(d["train_time"]/d["dt"])
    count, total, squares = 0, 0.0, 0.0
    for shard in shards:
        if shard["split"] != "train":
            continue
        with h5py.File(folder/shard["file"], "r") as f:
            for trajectory in f[f"u_{d['base_n']}"]:
                a = trajectory[:train_steps+1].astype(np.float64)
                count += a.size
                total += float(a.sum())
                squares += float(np.square(a).sum())
    mean = total/count if count else None
    norm = {"mean": mean, "std": float(np.sqrt(squares/count-mean**2)) if count else None, "count": count,
            "split": "train" if count else "none-evaluation-only", "resolution": d["base_n"], "end_time": d["train_time"]}
    manifest = {"data_signature": data_signature(cfg), "fixture": cfg["fixture"], "config": cfg,
                "shards": shards, "normalization": norm,
                "solver_verification_hash": file_hash(root/"verification"/"solver.json"),
                "environment": environment()}
    manifest["dataset_hash"] = digest({"signature": data_signature(cfg), "normalization": norm,
                                      "shards": {s["file"]:s["sha256"] for s in shards}})
    write_json(folder/"manifest.json", manifest)
    if cfg.get("evaluation_only", False):
        write_json(root/"verification"/"stress-data.json", {
            "purpose":"Measure sharp gradients/high-mode energy; do not assume every low-viscosity sample forms a sharp front",
            "viscosity":d["viscosity"],"dataset_hash":manifest["dataset_hash"],
            "records":[r for shard in shards for r in shard.get("shock_diagnostics", [])]})
    (folder/"dataset-card.md").write_text(
        "# Dataset card\n\n" + ("**Integration fixture; not research evidence.**\n\n" if cfg["fixture"] else "") +
        f"Protocol: `{cfg['protocol']}`. Dataset hash: `{manifest['dataset_hash']}`.\n\n"
        f"Counts: {d['counts']}. Periodic domain [0,1), viscosity {d['viscosity']}. "
        f"Reference resolution {d['reference_n']}; model grids {d['resolutions']}. "
        f"Time interval [0,{d['final_time']}], output dt={d['dt']}.\n\n"
        f"Training normalization: mean={norm['mean']}, std={norm['std']}, "
        f"fitted only to training N={d['base_n']} snapshots through t={d['train_time']}.\n\n"
        "Initial conditions are independent truncated periodic Gaussian Fourier fields plus "
        "a uniform random mean. See the recorded configuration and numerical-method documentation "
        "for coefficient variances and solver verification. Whole trajectories are assigned "
        "to immutable splits before generation. Higher grids are Fourier projections of "
        "the same reference solution.\n\n"
        "Limitations: fixed viscosity and initial-condition distribution; numerical reference "
        "and projection error remain. No external data are used. Solver checks, projection "
        "errors, seed IDs, shard hashes, and environment are in manifest.json and splits.json.\n")
    return manifest


def validate(cfg, root):
    root = Path(root)
    manifest = read_json(root/"data"/"manifest.json")
    if manifest["data_signature"] != data_signature(cfg):
        raise ValueError("Dataset configuration mismatch")
    expected_hash = digest({"signature": data_signature(cfg), "normalization": manifest["normalization"],
                           "shards": {s["file"]:s["sha256"] for s in manifest["shards"]}})
    if expected_hash != manifest["dataset_hash"]:
        raise ValueError("Dataset manifest hash mismatch")
    expected = split_manifest(cfg)
    if read_json(root/"data"/"splits.json")["trajectories"] != expected:
        raise ValueError("Split manifest mismatch")
    seen, seeds = [], []
    for info in manifest["shards"]:
        path = root/"data"/info["file"]
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with h5py.File(path, "r") as f:
            d = cfg["data"]
            times = np.arange(round(d["final_time"]/d["dt"])+1)*d["dt"]
            np.testing.assert_array_equal(f["times"][:],times)
            assert f.attrs["data_signature"] == data_signature(cfg)
            assert f.attrs["viscosity"] == d["viscosity"]
            for n in d["resolutions"]:
                assert f[f"u_{n}"].shape == (len(info["ids"]),len(times),n)
                assert f[f"u_{n}"].dtype == np.dtype("float32")
                np.testing.assert_array_equal(f[f"x_{n}"][:],np.arange(n)/n)
            assert f["reference"].shape == (len(info["ids"]),len(times),d["reference_n"])
            assert f["reference"].dtype == np.dtype("float64")
            ids = [s.decode() for s in f["ids"][:]]
            if ids != info["ids"]:
                raise ValueError("Trajectory IDs do not match manifest")
            seen.extend(ids)
            seeds.extend(int(v) for v in f["seeds"][:])
            for key in ["reference"] + [f"u_{n}" for n in cfg["data"]["resolutions"]]:
                for a in f[key]:
                    if not np.isfinite(a).all():
                        raise ValueError(f"Nonfinite values: {path}/{key}")
    assert seen == [r["id"] for r in expected]
    assert seeds == [r["seed"] for r in expected] and len(set(seeds)) == len(seeds)
    result = {"passed": True, "dataset_hash": manifest["dataset_hash"], "trajectories": len(seen),
              "manifest_sha256": file_hash(root/"data"/"manifest.json")}
    write_json(root/"verification"/"data.json", result)
    return result


def training_view(cfg, root, manifest):
    """Deterministic train-only subset and normalization, shared by all seeds.

    Source dataset hashes remain unchanged. This separately hashed view is saved
    with checkpoints so subset membership/normalization are independently auditable.
    """
    if cfg.get("evaluation_only", False):
        return None
    rows = [(s["file"], i, tid) for s in manifest["shards"] if s["split"] == "train"
            for i, tid in enumerate(s["ids"])]
    count = cfg["training"].get("train_trajectories", len(rows))
    if not 0 < count <= len(rows):
        raise ValueError("Invalid training subset size")
    if count == len(rows):
        ids = [r[2] for r in rows]
        view = {"dataset_hash": manifest["dataset_hash"], "ids": ids, "normalization": manifest["normalization"]}
        return {**view, "hash": digest(view)}
    indices = np.sort(np.random.default_rng(cfg["training"].get("subset_seed", 1729)).choice(len(rows), count, replace=False))
    selected = [rows[i] for i in indices]
    ids = [r[2] for r in selected]
    path = Path(root)/"training-view.json"
    if path.exists():
        saved = read_json(path)
        view = {k: v for k, v in saved.items() if k != "hash"}
        if saved["hash"] != digest(view) or saved["dataset_hash"] != manifest["dataset_hash"] or saved["ids"] != ids:
            raise ValueError("Training view changed; use a new work directory")
        return saved
    total = squares = 0.0
    elements = 0
    steps = round(cfg["data"]["train_time"]/cfg["data"]["dt"])
    for filename, index, _ in selected:
        with h5py.File(Path(root)/"data"/filename, "r") as f:
            a = f[f"u_{cfg['data']['base_n']}"][index, :steps+1].astype(np.float64)
        total += float(a.sum()); squares += float((a*a).sum()); elements += a.size
    mean = total/elements
    norm = {"mean": mean, "std": float(np.sqrt(squares/elements-mean**2)), "count": elements,
            "split": "train-subset", "resolution": cfg["data"]["base_n"], "end_time": cfg["data"]["train_time"]}
    view = {"dataset_hash": manifest["dataset_hash"], "ids": ids, "normalization": norm}
    saved = {**view, "hash": digest(view)}
    write_json(path, saved)
    return saved


def reuse_dataset(cfg, root, source):
    """Attach immutable data to a fresh experiment, with independent verification metadata."""
    import shutil
    root, source = Path(root), Path(source).resolve()
    manifest = read_json(source/"data"/"manifest.json")
    if manifest["data_signature"] != data_signature(cfg):
        raise ValueError("Source dataset does not match this configuration")
    check = read_json(source/"verification"/"data.json")
    if not check["passed"] or check["manifest_sha256"] != file_hash(source/"data"/"manifest.json"):
        raise ValueError("Source dataset requires validation")
    root.mkdir(parents=True, exist_ok=True)
    target = root/"data"
    if target.exists() or target.is_symlink():
        if target.resolve() != (source/"data").resolve():
            raise ValueError("Refusing to replace existing dataset")
    else:
        target.symlink_to(source/"data", target_is_directory=True)
    (root/"verification").mkdir(exist_ok=True)
    for name in ("data.json", "solver.json"):
        destination = root/"verification"/name
        if destination.exists() and file_hash(destination) != file_hash(source/"verification"/name):
            raise ValueError("Existing verification differs")
        if not destination.exists():
            shutil.copy2(source/"verification"/name, destination)
    return training_view(cfg, root, manifest)


class Store:
    """Bounded HDF5 handle cache; no dataset-sized memory allocation."""
    def __init__(self, cfg, root, split, validated=True):
        self.folder = Path(root)/"data"
        self.manifest = read_json(self.folder/"manifest.json")
        if self.manifest["data_signature"] != data_signature(cfg):
            raise ValueError("Dataset configuration mismatch")
        if validated:
            check = read_json(Path(root)/"verification"/"data.json")
            if not check["passed"] or check["manifest_sha256"] != file_hash(self.folder/"manifest.json"):
                raise ValueError("Run validate-data before using this dataset")
        self.rows = [(s["file"],i,tid) for s in self.manifest["shards"] if s["split"] == split
                     for i,tid in enumerate(s["ids"])]
        self.view = training_view(cfg, root, self.manifest)
        self.normalization = self.view["normalization"] if self.view else None
        if split == "train" and self.view:
            selected = set(self.view["ids"])
            self.rows = [r for r in self.rows if r[2] in selected]
        if not self.rows:
            raise ValueError(f"No trajectories for {split}")
        self.handles = OrderedDict()
        self.datasets = {}
        self.max_handles = cfg["runtime"].get("hdf5_handles", 128)
        self.steps = round(cfg["data"]["train_time"]/cfg["data"]["dt"])
        self.base_n = cfg["data"]["base_n"]
        self.training_cache = None
        if split == "train" and cfg["runtime"].get("cache_training", False):
            self.training_cache = np.stack([self.trajectory(i, self.base_n, self.steps+1)
                                            for i in range(len(self.rows))])
            print(f"Cached {len(self.rows)} training trajectories: {self.training_cache.nbytes/1e6:.1f} MB", flush=True)

    def _handle(self, filename):
        if filename not in self.handles:
            self.handles[filename] = h5py.File(self.folder/filename, "r", rdcc_nbytes=1024*1024)
            self.datasets[filename] = {}
        self.handles.move_to_end(filename)
        while len(self.handles) > self.max_handles:
            name, f = self.handles.popitem(last=False)
            del self.datasets[name]
            f.close()
        return self.handles[filename]

    def _dataset(self, filename, n):
        handle = self._handle(filename)
        if n not in self.datasets[filename]:
            self.datasets[filename][n] = handle[f"u_{n}"]
        return self.datasets[filename][n]

    def trajectory(self, i, n, stop=None):
        filename, j, _ = self.rows[i]
        return self._dataset(filename,n)[j, :stop]

    def pairs(self, indices):
        if self.training_cache is not None:
            indices = np.asarray(indices)
            trajectories, steps = indices//self.steps, indices%self.steps
            return (self.training_cache[trajectories, steps, None, :],
                    self.training_cache[trajectories, steps+1, None, :])
        x, y = [], []
        for index in indices:
            trajectory, step = divmod(int(index),self.steps)
            filename,j,_ = self.rows[trajectory]
            pair = self._dataset(filename,self.base_n)[j,step:step+2]
            x.append(pair[0]); y.append(pair[1])
        return np.array(x)[:,None,:], np.array(y)[:,None,:]

    def close(self):
        for f in self.handles.values():
            f.close()
        self.handles.clear()
        self.datasets.clear()
        self.training_cache = None
