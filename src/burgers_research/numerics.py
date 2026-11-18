"""Float64 reference methods. Coordinates are x_j=j/N on the unit torus."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .common import data_signature, environment, write_json


def project(u, n):
    return resample(u, n, axis=-1) if u.shape[-1] != n else u.copy()


def initial_condition(n, seed, data):
    rng = np.random.default_rng(seed)
    k = np.arange(1, data["ic_modes"] + 1)
    variance = ((2 * np.pi * k) ** 2 + 25) ** -2
    # E[spatial mean(g^2)] = sum_k variance_k * scale^2.
    scale = data["ic_rms"] / np.sqrt(variance.sum())
    a, b = rng.normal(size=(2, len(k))) * np.sqrt(variance) * scale
    mean = rng.uniform(*data["mean_range"])
    angles = 2 * np.pi * np.arange(n)[:, None] / n * k[None, :]
    return mean + np.cos(angles) @ a + np.sin(angles) @ b


def derivative(u, order=1):
    n = u.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1 / n)
    factor = (1j * k) ** order
    if order % 2 and n % 2 == 0:
        factor[-1] = 0
    return np.fft.irfft(np.fft.rfft(u) * factor, n=n)


def nonlinear(v, n):
    u = np.fft.irfft(v, n=n)
    padded = project(u, 3 * n // 2)
    flux = project(0.5 * padded ** 2, n)
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1 / n)
    k[-1] = 0  # Real-valued Nyquist mode has zero first derivative at nodes.
    return -1j * k * np.fft.rfft(flux)


def coefficients(n, nu, h):
    linear = -nu * (2 * np.pi * np.fft.rfftfreq(n, d=1 / n)) ** 2
    E, E2 = np.exp(h * linear), np.exp(h * linear / 2)
    roots = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    z = h * linear[:, None] + roots
    Q = h * np.real(np.mean((np.exp(z / 2) - 1) / z, axis=1))
    f1 = h * np.real(np.mean((-4 - z + np.exp(z) * (4 - 3*z + z*z)) / z**3, axis=1))
    f2 = h * np.real(np.mean((2 + z + np.exp(z) * (-2 + z)) / z**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3*z - z*z + np.exp(z) * (4 - z)) / z**3, axis=1))
    return E, E2, Q, f1, f2, f3


def solve(u0, nu, output_dt, final_time, internal_dt):
    n = len(u0)
    substeps = int(np.ceil(output_dt / internal_dt - 1e-12))
    h = output_dt / substeps
    E, E2, Q, f1, f2, f3 = coefficients(n, nu, h)
    count = round(final_time / output_dt)
    out = np.empty((count + 1, n), dtype=np.float64)
    out[0] = u0
    v = np.fft.rfft(u0)
    for t in range(1, count + 1):
        for _ in range(substeps):
            Nv = nonlinear(v, n)
            a = E2 * v + Q * Nv
            Na = nonlinear(a, n)
            b = E2 * v + Q * Na
            Nb = nonlinear(b, n)
            c = E2 * a + Q * (2 * Nb - Nv)
            v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * nonlinear(c, n)
        out[t] = np.fft.irfft(v, n=n)
        if not np.isfinite(out[t]).all():
            raise FloatingPointError(f"Nonfinite reference solution at t={t * output_dt}")
    return out


def cole_hopf(n, t, nu, amplitude=0.2):
    x = np.arange(n) / n
    a = amplitude * np.exp(-nu * (2 * np.pi)**2 * t)
    return 4 * np.pi * nu * a * np.sin(2*np.pi*x) / (1 + a*np.cos(2*np.pi*x))


def physical_checks(u):
    scale = max(np.sqrt(np.mean(u[0]**2)), 1e-12)
    energy = 0.5 * np.mean(u**2, axis=-1)
    return {"mass_drift": float(np.max(np.abs(u.mean(-1) - u[0].mean())) / scale),
            "energy_increase": float(max(0, np.diff(energy).max()) / max(energy[0], 1e-24))}


def verify_solver(cfg, root):
    d, s = cfg["data"], cfg["solver"]
    start = time.perf_counter()
    n = d["reference_n"]
    dt = s["initial_dt"]
    history = []
    for refinement in range(s["max_refinements"] + 1):
        rows = []
        for seed in s["verification_seeds"]:
            try:
                coarse = solve(initial_condition(n, seed, d), d["viscosity"], d["dt"], d["final_time"], dt)
                finer_t = solve(initial_condition(n, seed, d), d["viscosity"], d["dt"], d["final_time"], dt/2)
                finer_x = solve(initial_condition(2*n, seed, d), d["viscosity"], d["dt"], d["final_time"], dt/2)
            except FloatingPointError:
                # Low-viscosity advection can need a smaller explicit timestep.
                # No spatial error estimate is available from a divergent solve.
                rows.append({"seed":seed, "temporal_error":float("inf"), "spatial_error":None,
                             "mass_drift":float("inf"), "energy_increase":float("inf"),
                             "failed_solve":True})
                continue
            scale = max(np.sqrt(np.mean(coarse[0]**2)), 1e-12)
            temporal = float(np.sqrt(np.mean((coarse-finer_t)**2, axis=-1)).max() / scale)
            spatial = float(np.sqrt(np.mean((finer_t-project(finer_x, n))**2, axis=-1)).max() / scale)
            rows.append({"seed": seed, "temporal_error": temporal, "spatial_error": spatial,
                         **physical_checks(coarse)})
        exact = cole_hopf(n, d["final_time"], d["viscosity"])
        analytic = solve(cole_hopf(n, 0, d["viscosity"]), d["viscosity"], d["dt"], d["final_time"], dt)[-1]
        analytic_error = float(np.linalg.norm(analytic-exact)/np.linalg.norm(exact))
        passed = analytic_error < s["convergence_tolerance"] and all(
            r["temporal_error"] < s["convergence_tolerance"] and
            r["spatial_error"] < s["convergence_tolerance"] and
            r["mass_drift"] < s["mass_tolerance"] and
            r["energy_increase"] < s["energy_tolerance"] for r in rows)
        history.append({"internal_dt": dt, "rows": rows, "analytic_error": analytic_error})
        print(f"Solver refinement {refinement}: dt={dt:g}, passed={passed}", flush=True)
        if passed:
            break
        if any(r["spatial_error"] is not None and r["spatial_error"] >= s["convergence_tolerance"] for r in rows):
            break  # Time refinement cannot fix insufficient spatial resolution.
        dt /= 2
    result = {"passed": passed, "data_signature": data_signature(cfg), "internal_dt": dt,
              "history": history, "seconds": time.perf_counter()-start, "environment": environment()}
    write_json(Path(root)/"verification"/"solver.json", result)
    if not passed:
        raise RuntimeError("Solver verification failed. Refine solver configuration; do not generate production data.")
    return result
