import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width,width,modes,dtype=torch.cfloat)/(width*width))

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=-1)
        if spectrum.shape[-1] < self.modes:
            raise ValueError("Grid too small for retained Fourier modes")
        low = torch.einsum("bik,iok->bok", spectrum[...,:self.modes], self.weight)
        out = F.pad(low, (0,spectrum.shape[-1]-self.modes))
        return torch.fft.irfft(out,n=x.shape[-1],dim=-1)


class FNO1d(nn.Module):
    def __init__(self, width=64, modes=16, layers=4, projection=128):
        super().__init__()
        self.lift = nn.Conv1d(1,width,1)
        self.spectral = nn.ModuleList(SpectralConv1d(width,modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv1d(width,width,1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv1d(width,projection,1),nn.GELU(),nn.Conv1d(projection,1,1))

    def forward(self,x):
        x = self.lift(x)
        for i,(spectral,local) in enumerate(zip(self.spectral,self.local)):
            x = spectral(x) + local(x)
            if i < len(self.spectral)-1:
                x = F.gelu(x)
        return self.project(x)


def double_conv(cin,cout):
    return nn.Sequential(nn.Conv1d(cin,cout,3,padding=1,padding_mode="circular"),nn.ReLU(),
                         nn.Conv1d(cout,cout,3,padding=1,padding_mode="circular"),nn.ReLU())


class UNet1d(nn.Module):
    def __init__(self,channels=(32,64,128,256)):
        super().__init__()
        self.encoders = nn.ModuleList(double_conv(a,b) for a,b in zip([1]+list(channels[:-1]),channels))
        self.ups = nn.ModuleList(nn.ConvTranspose1d(a,b,2,stride=2) for a,b in zip(channels[:0:-1],channels[-2::-1]))
        self.decoders = nn.ModuleList(double_conv(2*c,c) for c in channels[-2::-1])
        self.output = nn.Conv1d(channels[0],1,1)

    def forward(self,x):
        if x.shape[-1] % 2**(len(self.encoders)-1):
            raise ValueError("U-Net grid must be divisible by its total downsampling factor")
        skips = []
        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders)-1:
                skips.append(x)
                x = F.max_pool1d(x,2)
        for up,decoder,skip in zip(self.ups,self.decoders,reversed(skips)):
            x = decoder(torch.cat((up(x),skip),dim=1))
        return self.output(x)


class TimeStepper(nn.Module):
    """Normalized-state update; linear normalization preserves mean projection.

    Residual outputs are normalized time derivatives, multiplied by physical dt.
    Projection conserves the spatial mean algebraically, up to floating point.
    """
    def __init__(self, network, dt, stepping="residual", conserve_mean=False):
        super().__init__()
        self.network = network
        self.dt = dt
        self.stepping = stepping
        self.conserve_mean = conserve_mean

    def forward(self, x):
        raw = self.network(x)
        update = self.dt * raw if self.stepping == "residual" else raw - x
        if self.conserve_mean:
            update = update - update.mean(dim=-1, keepdim=True)
        return x + update


def build_model(kind,cfg,spec=None):
    if kind == "fno":
        model = FNO1d(**cfg["model"]["fno"])
    elif kind == "unet":
        model = UNet1d(**cfg["model"]["unet"])
    else:
        raise ValueError(f"Unknown architecture: {kind}")
    spec = spec or {}
    stepping = spec.get("stepping", "direct")
    if stepping not in ("direct", "residual"):
        raise ValueError(f"Unknown stepping: {stepping}")
    if stepping != "direct" or spec.get("conserve_mean", False):
        model = TimeStepper(model, cfg["data"]["dt"], stepping, spec.get("conserve_mean", False))
    return model


def parameter_count(model):
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
