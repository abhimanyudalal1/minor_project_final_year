import numpy as np
import pytest
import torch
from burgers_research.numerics import cole_hopf, derivative, initial_condition, physical_checks, project, solve
from burgers_research.physics import residual, resize, spatial_terms

def test_solver_analytic_and_constant():
    n,nu = 128,0.01
    np.testing.assert_allclose(solve(np.full(n,0.7),nu,0.01,0.1,0.001),0.7,atol=1e-14)
    actual = solve(cole_hopf(n,0,nu),nu,0.01,0.1,0.001)
    np.testing.assert_allclose(actual[-1],cole_hopf(n,0.1,nu),atol=1e-11)
    assert physical_checks(actual)["mass_drift"] < 1e-12
    assert physical_checks(actual)["energy_increase"] == 0

def test_time_convergence(cfg):
    u = initial_condition(256,13,cfg["data"])
    reference = solve(u,0.01,0.01,0.04,0.000125)[-1]
    coarse = solve(u,0.01,0.01,0.04,0.002)[-1]
    fine = solve(u,0.01,0.01,0.04,0.001)[-1]
    assert np.linalg.norm(fine-reference) < np.linalg.norm(coarse-reference)/8

@pytest.mark.parametrize("n,new",[(32,48),(48,32),(32,64),(64,32),(32,32)])
def test_resize_matches_scipy_including_nyquist(n,new):
    u = np.random.default_rng(5).normal(size=(2,1,n))
    actual = resize(torch.from_numpy(u),new).numpy()
    np.testing.assert_allclose(actual,project(u,new),atol=1e-12)
    np.testing.assert_allclose(actual.mean(-1),u.mean(-1),atol=1e-12)

def test_spectral_derivative():
    x = np.arange(64)/64
    np.testing.assert_allclose(derivative(np.sin(6*np.pi*x)),6*np.pi*np.cos(6*np.pi*x),atol=1e-12)

@pytest.mark.parametrize("method",["spectral","finite_difference"])
def test_residual_mass_identity_and_differentiability(method):
    rng = np.random.default_rng(4)
    u = torch.tensor(rng.normal(size=(2,1,64)),requires_grad=True)
    v = torch.tensor(rng.normal(size=(2,1,64)),requires_grad=True)
    r = residual(u,v,0.01,0.01,method)
    torch.testing.assert_close(r.mean(-1),(v-u).mean(-1)/0.01,rtol=1e-10,atol=1e-10)
    r.square().mean().backward()
    assert torch.isfinite(v.grad).all()
    flux,diffusion = spatial_terms(u,method)
    assert flux.sum(-1).abs().max() < 1e-10
    assert diffusion.sum(-1).abs().max() < 1e-9

def test_finite_difference_convergence():
    errors = []
    for n in (64,128):
        x = torch.arange(n,dtype=torch.float64)/n
        u = torch.sin(2*np.pi*x)
        _,second = spatial_terms(u,"finite_difference")
        errors.append(float((second+(2*np.pi)**2*u).abs().max()))
    assert errors[0]/errors[1] > 3.9
