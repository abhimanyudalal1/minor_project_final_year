# Numerical methods and dataset provenance

## Solver and verification

The solver uses double-precision real FFTs on `x_j=j/N`, with no repeated
endpoint. Nonlinearity is the conservative flux derivative `-D_x(u²/2)`.
Products are evaluated on a grid padded to 3N/2 and projected back. SciPy Fourier
resampling handles even-grid Nyquist splitting; odd spectral derivatives have
zero Nyquist multiplier. Viscosity is diagonal in Fourier space.

Time integration uses Kassam–Trefethen ETDRK4 with a 32-point complex contour
average for stable coefficients near zero. Each output interval divides into
an integer number of internal steps. Verification starts at 0.001 and halves
the step if temporal accuracy fails. Spatial failure requires a configuration
revision; it cannot be fixed by pretending time refinement improves the grid.

Compare h and h/2 at N=8192, then N=8192 and 16384 at h/2 with identical analytic
initial fields and four development seeds. Maximum snapshot RMS differences,
normalized by initial RMS, must be <1e-5. Mass drift must be <1e-10 of initial
RMS; positive energy increments must be <1e-10 of initial energy. An analytic
periodic Cole–Hopf solution independently checks implementation accuracy.
Tests also cover constants, derivative identities, and convergence order.

These checks establish accuracy on the verification set, not a proof for every
Gaussian realization. Every production trajectory additionally checks finite
values, mass, and energy. A failing seed is retained for investigation.

## Storage schema and generation

Each HDF5 shard holds up to 16 trajectories:

- `ids`, `seeds`, `times`, `x_reference`, and `x_<N>`;
- `reference`: `[trajectory, time, 8192]`, float64;
- `u_256`, `u_512`, `u_1024`: `[trajectory, time, space]`, float32;
- viscosity, internal timestep, and data-signature attributes.

Reference chunks hold one trajectory and snapshot; model chunks hold up to 16
snapshots of one trajectory. LZF compression is lossless. Generation keeps one
reference trajectory in memory. Atomic shard rename and checksums distinguish
complete shards; interrupted partial shards are regenerated with the same seeds.
Finished shards are verified and skipped on resume.

`splits.json` records IDs and seeds before generation. All grids share IDs.
`manifest.json` links shard hashes, physics checks, projection errors, training
normalization, solver verification, and environment. Loaders never resplit data.

Fit one global mean and population standard deviation from the 101 training
snapshots at N=256, including both endpoints. Validation/test data, higher grids,
and times after t=1 never affect normalization. Pair loading uses a bounded HDF5
handle cache. Fourier projection preserves the zero mode; projection error is
reported separately from solver error.

## Physics loss and error floors

For physical input u and predicted v:

`R=(v-u)/dt + [D_x(u²/2)+D_x(v²/2)]/2 - nu[D_xx(u)+D_xx(v)]/2`.

The objective is normalized supervised MSE plus
`lambda * mean((dt * R / training_std)²)`. At zero lambda physics is not computed.
Spectral and second-order centered periodic finite differences share this
trapezoidal time rule. The latter differentiates flux, not `u*D_x(u)`.
Both derivative sums vanish, giving `mean(R)=[mean(v)-mean(u)]/dt`.

This residual is approximate even on correct reference trajectories. Reference
controls report its error floor on each grid, including float32 storage and
projection effects. Energy diagnostics use the trapezoidal integral of viscous
dissipation. No result equates soft residual satisfaction with exact conservation.

Source: [Kassam & Trefethen, 2005](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf).
