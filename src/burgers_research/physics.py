import numpy as np
import torch
from torch.nn import functional as F


def resize(u,n):
    old = u.shape[-1]
    if old == n:
        return u
    spectrum = torch.fft.rfft(u,norm="forward")
    count = min(spectrum.shape[-1],n//2+1)
    out = F.pad(spectrum[...,:count],(0,n//2+1-count))
    # Split/merge the self-conjugate Nyquist coefficient, matching scipy.resample.
    if old % 2 == 0 and n > old:
        scale = torch.ones(n//2+1,device=u.device,dtype=u.dtype)
        scale[old//2] = 0.5
        out = out * scale
    if n % 2 == 0 and n < old:
        out = torch.cat((out[...,:-1],2*out[...,-1:].real.to(out.dtype)),dim=-1)
    return torch.fft.irfft(out,n=n,norm="forward")


def spectral_derivative(u,order):
    n = u.shape[-1]
    k = 2*np.pi*torch.fft.rfftfreq(n,d=1/n,device=u.device,dtype=u.dtype)
    multiplier = (1j*k)**order
    if order % 2 == 1 and n % 2 == 0:
        multiplier[-1] = 0
    return torch.fft.irfft(torch.fft.rfft(u)*multiplier,n=n)


def spatial_terms(u,method):
    n = u.shape[-1]
    if method == "spectral":
        padded = resize(u,3*n//2)
        flux = resize(padded.square()/2,n)
        return spectral_derivative(flux,1), spectral_derivative(u,2)
    if method == "finite_difference":
        flux = u.square()/2
        first = (torch.roll(flux,-1,-1)-torch.roll(flux,1,-1))*(n/2)
        second = (torch.roll(u,-1,-1)-2*u+torch.roll(u,1,-1))*n*n
        return first,second
    raise ValueError(f"Unknown derivative: {method}")


def residual(u,v,dt,nu,method="spectral"):
    fu,du = spatial_terms(u,method)
    fv,dv = spatial_terms(v,method)
    return (v-u)/dt+(fu+fv)/2-nu*(du+dv)/2


def loss(prediction,target,physical_input,norm,dt,nu,weight=0.0,method="spectral"):
    data = (prediction-target).square().mean()
    if weight == 0:
        return data, data.detach(), data.detach().new_zeros(())
    physical_prediction = prediction*norm["std"]+norm["mean"]
    pde = (dt*residual(physical_input,physical_prediction,dt,nu,method)/norm["std"]).square().mean()
    return data+weight*pde, data.detach(),pde.detach()


def training_loss(model, physical_input, physical_target, norm, dt, nu, spec):
    """One supervised step plus mean physics loss over one/two predicted steps.

    Two-step mode backpropagates through BOTH predictions (no detach), and needs
    no second target. Thus every original adjacent training pair is retained.
    """
    z = (physical_input - norm["mean"]) / norm["std"]
    prediction = model(z)
    target = (physical_target - norm["mean"]) / norm["std"]
    weight = spec["weight"]
    steps = spec.get("physics_steps", 1)
    if steps not in (1, 2):
        raise ValueError("physics_steps must be 1 or 2")
    if weight and steps == 2:
        first = prediction * norm["std"] + norm["mean"]
        second = model(prediction) * norm["std"] + norm["mean"]
        r1 = residual(physical_input, first, dt, nu, spec["derivative"])
        r2 = residual(first, second, dt, nu, spec["derivative"])
        physics = 0.5 * ((dt*r1/norm["std"]).square().mean() + (dt*r2/norm["std"]).square().mean())
        supervised = (prediction-target).square().mean()
        return supervised + weight*physics, supervised.detach(), physics.detach()
    return loss(prediction, target, physical_input, norm, dt, nu, weight, spec["derivative"])
