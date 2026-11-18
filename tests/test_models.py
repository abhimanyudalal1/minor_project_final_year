import copy
import pytest
import torch
from burgers_research.common import matrix, seed_all
from burgers_research.models import build_model
from burgers_research.physics import loss

@pytest.mark.parametrize("kind",["fno","unet"])
@pytest.mark.parametrize("n",[256,512,1024])
def test_shapes_and_backward(cfg,kind,n):
    from burgers_research.common import load_config
    cfg = load_config("configs/study.yaml")
    model = build_model(kind,cfg)
    x = torch.randn(2,1,n)
    y = model(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_accumulation_and_zero_weight(cfg):
    seed_all(1)
    a = build_model("fno",cfg); b = copy.deepcopy(a)
    x,y = torch.randn(7,1,32),torch.randn(7,1,32)
    norm = {"mean":0.0,"std":1.0}
    full,_,pde = loss(a(x),y,x,norm,0.01,0.01,0)
    assert pde == 0
    full.backward()
    for start in range(0,7,3):
        sub = x[start:start+3]
        part,_,_ = loss(b(sub),y[start:start+3],sub,norm,0.01,0.01,0)
        (part*len(sub)/len(x)).backward()
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-5)

def test_production_matrix():
    from burgers_research.common import load_config
    runs = list(matrix(load_config("configs/study.yaml")))
    assert len(runs) == 50 and len({r["name"] for r in runs}) == 50
    for seed in range(5):
        assert len([r for r in runs if r["seed"] == seed]) == 10

def test_gradient_guard(cfg):
    from burgers_research.common import gradients_finite
    model = build_model("fno",cfg)
    model(torch.randn(2,1,32)).square().mean().backward()
    assert gradients_finite(model)
    next(model.parameters()).grad.flatten()[0] = float("nan")
    assert not gradients_finite(model)

def test_spectral_layer_matches_explicit_mode_matrices():
    from burgers_research.models import SpectralConv1d
    torch.manual_seed(19)
    layer = SpectralConv1d(8, 6)
    x = torch.randn(2, 8, 32, requires_grad=True)
    actual = layer(x)
    spectrum = torch.fft.rfft(x)
    modes = [spectrum[..., k] @ layer.weight[..., k] if k < 6
             else torch.zeros_like(spectrum[..., k]) for k in range(spectrum.shape[-1])]
    reference = torch.fft.irfft(torch.stack(modes, dim=-1), n=32)
    torch.testing.assert_close(actual, reference)
    grad_actual = torch.autograd.grad(actual.square().sum(), (x, layer.weight), retain_graph=True)
    grad_reference = torch.autograd.grad(reference.square().sum(), (x, layer.weight))
    for a, b in zip(grad_actual, grad_reference):
        torch.testing.assert_close(a, b)
