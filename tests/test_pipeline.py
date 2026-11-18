import numpy as np
import pytest
import torch
from burgers_research.common import matrix
from burgers_research.data import Store, generate, split_manifest, validate
from burgers_research.evaluation import diagnostics, evaluate, evaluate_controls, interpolate
from burgers_research.numerics import verify_solver
from burgers_research.reporting import bootstrap_difference, build_report, summarize
from burgers_research.training import lock_selection, train

def test_interpolation_periodic():
    u = np.sin(2*np.pi*np.arange(32)/32)[None,:]
    for kind in ("linear","cubic"):
        out = interpolate(u,128,kind)
        np.testing.assert_allclose(out[:,::4],u,atol=1e-12)
        assert abs(out[0,-1]-np.sin(2*np.pi*127/128)) < 0.001

def test_failure_and_metric_identity(cfg):
    x = np.arange(32)/32
    truth = np.repeat((0.3+np.sin(2*np.pi*x))[None,:],5,axis=0)
    exact = diagnostics(truth,truth,cfg)
    assert exact["horizons"]["4"]["rollout_error"] == 0
    assert exact["horizons"]["4"]["max_mass_drift"] == 0
    broken = truth.copy(); broken[2:] = np.nan
    bad = diagnostics(broken,truth,cfg)
    assert bad["horizons"]["4"]["failed"]
    assert np.isinf(bad["horizons"]["4"]["rollout_error"])

def test_paired_bootstrap():
    a = np.arange(12).reshape(3,4)
    result = bootstrap_difference(a+2,a,200,1)
    assert result["difference"] == 2 and result["ci95"] == [2,2]

def test_end_to_end_and_restart(cfg,tmp_path):
    verify_solver(cfg,tmp_path)
    manifest = generate(cfg,tmp_path)
    assert generate(cfg,tmp_path)["dataset_hash"] == manifest["dataset_hash"]
    validate(cfg,tmp_path)
    splits = split_manifest(cfg)
    assert len({r["seed"] for r in splits}) == len(splits)
    store = Store(cfg,tmp_path,"train")
    store.max_handles = 1  # Exercise eviction and reopening of cached HDF5 datasets.
    raw = np.stack([store.trajectory(i,32,3) for i in range(4)]).astype(float)
    np.testing.assert_allclose(manifest["normalization"]["mean"],raw.mean(),atol=1e-12)
    np.testing.assert_allclose(manifest["normalization"]["std"],raw.std(),atol=1e-12)
    assert len(store.handles) == 1 and len(store.datasets) == 1
    store.close()
    with pytest.raises(ValueError,match="registered training matrix"):
        lock_selection(cfg,tmp_path)
    runs = list(matrix(cfg)); first = runs[1]
    train(cfg,tmp_path,first,max_updates=1)
    train(cfg,tmp_path,first)
    resumed = torch.load(tmp_path/"runs"/first["name"]/"last.pt",weights_only=False)
    alternate = dict(first,name="resume-reference")
    train(cfg,tmp_path,alternate)
    uninterrupted = torch.load(tmp_path/"runs"/alternate["name"]/"last.pt",weights_only=False)
    for key,a in resumed["model"].items():
        torch.testing.assert_close(a,uninterrupted["model"][key],rtol=0,atol=0)
    for spec in runs:
        train(cfg,tmp_path,spec)
    lock_selection(cfg,tmp_path)
    for spec in runs:
        evaluate(cfg,tmp_path,spec)
    evaluate_controls(cfg,tmp_path)
    summary = summarize(cfg,tmp_path)
    assert summary["fixture"]
    report = build_report(cfg,tmp_path)
    assert "not scientific evidence" in report.read_text()
    assert (report.parent/"relative_l2.png").exists()
    before = report.read_text()
    assert build_report(cfg,tmp_path).read_text() == before
    shard = tmp_path/"data"/manifest["shards"][0]["file"]
    with open(shard,"ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError,match="Checksum"):
        validate(cfg,tmp_path)
