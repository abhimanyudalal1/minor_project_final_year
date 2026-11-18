Reproducible research pipeline for physics-constrained neural operators
1. Research objective and fixed protocol
Build a documented, reproducible study of whether adding a soft PDE constraint improves rollout accuracy, conservation behavior, stability, and resolution generalization for periodic 1D viscous Burgers’ equation.

The deliverables are an installable Python project, validated dataset generator, standard model implementations, reproducible experiments, and a research report with automatically generated figures and tables.

Agreed requirements:
- Preserve the full dataset and experiment matrix across hardware platforms.
- Use five training seeds.
- Select checkpoints and physics weights using validation rollout accuracy.
- Include physics-weight sensitivity and spectral versus finite-difference residuals.
- Support the M4 Mac with 16 GB memory, with portable CUDA execution for Colab or Kaggle.
Research boundaries:
- Compare supervised U-Net, supervised FNO, and the identical FNO with a physics penalty.
- Keep viscosity fixed; viscosity shifts and additional architecture families are outside this study.
- Train exclusively at resolution 256, including physics-loss evaluation.
- Describe the physics model as PINO-style: the original PINO includes capabilities such as higher-resolution physics supervision that this controlled experiment deliberately excludes. PINO paper
- Correct the README’s formatting, unsupported novelty claims, “exact ground truth” terminology, and bibliography. The cited Cardoso-Bihlo reference should be replaced with a verified, relevant source. Verified conservation paper
Freeze the protocol before production training. Record subsequent changes with reasons and identify affected runs.
2. Dataset generation and numerical verification
Physical problem and dataset defaults
Use
\[
u_t+\partial_x(u^2/2)=\nu u_{xx},
\qquad x\in[0,1),\qquad \nu=0.01,
\]
with periodic boundaries and no duplicated endpoint.
Item	Planned specification
Independent trajectories	2,000 training, 250 validation, 500 test
Reference spatial grid	8,192 points
Model grids	256, 512, 1,024
Stored times	\(t=0\) through \(2\), spaced by \(0.01\)
Training interval	\(0\leq t\leq1\): 100 adjacent pairs per training trajectory
Validation selection	100-step rollout from \(t=0\), at resolution 256
Final testing	100-step and 200-step rollouts at all three resolutions


Generate smooth periodic initial conditions from a Fourier Gaussian random field: independent sine/cosine coefficients for modes 1–64, with variance proportional to \(((2\pi k)^2+25)^{-2}\), scaled to expected fluctuation RMS 0.5. Add an independent mean sampled uniformly from \([-0.5,0.5]\). Document this as a project-specific benchmark distribution; varying means makes conservation testing more informative.

Assign trajectory IDs, random seeds, and splits before generation. Every resolution and time window belonging to an initial condition must remain in its assigned split.
Reference solver
- Implement a CPU float64 Fourier pseudo-spectral solver in conservative flux form, with 3/2 dealiasing.
- Use the established ETDRK4 scheme to handle viscous stiffness. Kassam–Trefethen method
- Begin with internal timestep \(10^{-3}\); halve it until temporal convergence passes on a separate verification set. Compare spatial resolutions 8,192 and 16,384.
- Require normalized solution differences below \(10^{-5}\), mass drift below \(10^{-10}\) relative to the initial RMS scale, and energy monotonicity within numerical tolerance.
- If verification fails, refine the numerical settings before generating the production dataset. Do not compensate by loosening accuracy criteria.
- Generate lower-resolution reference fields through Fourier projection, preserving the zero mode. Quantify projection error separately from solver error.
Validate constants, analytic periodic Cole–Hopf solutions, derivative identities, convergence, mass conservation, and
\[
\frac{d}{dt}\frac12\int u^2\,dx=-\nu\int u_x^2\,dx.
\]
Storage and documentation
Use resumable HDF5 shards with trajectory-major chunks. Archive reference solutions in float64 and model-ready grids in float32. Store coordinates, viscosity, timestamps, initial-condition seeds, solver settings, validation results, and checksums.

Fit one global normalization mean and standard deviation using only training data at resolution 256 during \(t\leq1\). Compute physics losses and diagnostics after restoring physical units.

Stage deliverables: numerical-method documentation, convergence report, dataset card, immutable split manifest, and dataset integrity report.
3. Standard architectures and controlled training
Implement models in plain PyTorch, with citations and explicit documentation of adaptations. Both architectures accept normalized fields shaped [batch, 1, space] and predict the next field with the same shape.
Model	Architecture	Training objective
1D U-Net	Encoder channels 32/64/128/256; two kernel-3 convolutions and ReLU per level; circular padding; max pooling; transposed-convolution upsampling; concatenated skip connections; linear output	Supervised MSE
Vanilla FNO	Pointwise lift to 64 channels; four Fourier blocks retaining 16 nonnegative modes; spectral convolution plus pointwise linear path; GELU between blocks; projection through 128 channels to one output channel	Supervised MSE
Physics-constrained FNO	Exactly the same implementation and parameterization as vanilla FNO	MSE plus weighted PDE residual


The U-Net is a documented 1D periodic adaptation of the encoder–decoder architecture. The FNO retains the standard spectral structure; its next-step target, activation choice, and omission of batch normalization are documented adaptations. Avoid batch-dependent normalization so gradient accumulation preserves the intended objective. Report actual parameter counts without claiming exact capacity matching. U-Net, FNO

Do not add conservation projection or post-prediction correction to the core models.
Physics loss
For input \(u\) and prediction \(v\), use the trapezoidal discrete residual
\[
R(u,v)=\frac{v-u}{\Delta t}
+\frac12\left[D_x(u^2/2)+D_x(v^2/2)\right]
-\frac{\nu}{2}\left[D_{xx}u+D_{xx}v\right].
\]
Use
\[
L=L_{\mathrm{data}}+
\lambda\,\mathrm{mean}\left[\left(\frac{\Delta t\,R}{\sigma_{\mathrm{train}}}\right)^2\right].
\]
Compare spectral derivatives with dealiasing against second-order periodic finite differences, keeping temporal discretization unchanged. Evaluate both residuals on reference trajectories to expose discretization error floors.
Experiment matrix
- U-Net: five seeds.
- Vanilla FNO: five seeds.
- Physics FNO: two derivative methods × four weights \(\{0.1,1,10,100\}\) × five seeds.
- Total: 50 production training runs.
Use seeds 0–4, pairing FNO initialization and training-example order across vanilla and physics variants.

Train on every training pair each epoch using Adam, initial learning rate \(10^{-3}\), halving every 100 epochs, for 500 epochs. Use effective batch size 256, float32 training, no weight decay, and no augmentation. Smaller microbatches may use correctly weighted gradient accumulation.

Evaluate validation rollouts every ten epochs and retain the checkpoint with lowest mean time-averaged relative \(L^2\) error. Choose one weight per derivative method using the mean validation score across all five seeds; never select seeds. Preserve the complete sensitivity results.

Save model, optimizer, scheduler, normalization, RNG and sampler state, configuration, dataset hash, and environment information. Resume interrupted runs without changing their training budget.

Stage deliverables: architecture diagrams and shape tables, loss derivation, training protocol, experiment registry, learning curves, and per-run records.
4. Evaluation, controls, and statistical reporting
Initialize each rollout from the reference initial condition and subsequently feed predictions back into the model. Evaluate all models on the same held-out trajectories and physical timestamps.
Required measurements
- One-step MSE and relative \(L^2\) error.
- Rollout error curves, time-averaged error, and terminal error.
- Absolute mass drift and drift normalized by initial RMS, avoiding division by potentially zero initial mass.
- Energy error, increases in energy, and integrated dissipation-balance error.
- Spectral and finite-difference residuals for every model, regardless of training loss.
- Nonfinite predictions, divergent rollouts, and failure rates.
- Training time, inference time, peak memory, parameter count, and dataset-generation cost.
Use a documented denominator floor for near-zero reference norms. Failed rollouts must remain visible in aggregate reporting; do not silently average only successful cases.

Explain that residual and conservation diagnostics are separately measured but mathematically related: for the periodic conservative discretization, the spatial mean residual corresponds to the one-step mass defect. Study how these defects accumulate over time.
Resolution evaluation
- Run FNO directly at 256, 512, and 1,024 using unchanged weights and physical timestep.
- For the primary U-Net adapter, restrict the current field to 256 using Fourier projection, predict one step, and periodically interpolate back to the target grid at every rollout step.
- Use periodic cubic interpolation as the primary adapter and linear interpolation as a sensitivity check.
- Add evaluation-only controls: native higher-resolution U-Net, base-grid FNO plus interpolation, persistence, and interpolation of coarse reference trajectories. These require no additional training.
These controls separate model error from interpolation error and prevent unsupported claims that increasing resolution necessarily recovers missing information.
Statistical analysis and report
Report each seed, mean and standard deviation, and paired differences between vanilla and physics FNO. Use paired hierarchical bootstrap confidence intervals over seeds and trajectories; never treat adjacent timesteps as independent samples.

Generate tables and figures directly from saved metrics: rollout curves, mass drift, energy balance, residual-versus-drift plots, resolution comparisons, physics-weight sensitivity, and accuracy-versus-cost plots.

Write the report around research questions, related work, numerical validation, dataset construction, architectures, experimental protocol, results, and limitations. Negative or mixed findings are valid outcomes. Keep claims limited to the tested viscosity, initial-condition distribution, and horizons.
5. Pipeline interfaces, verification, and completion criteria
Reproducible project structure
Use an installable src/burgers_research/ package, versioned YAML configurations, automated tests, and Markdown documentation. Separate numerical methods, data access, models, losses, training, evaluation, and reporting.

Expose a common command-line workflow:

doctor → verify-solver → generate-data → validate-data → train → evaluate → summarize → build-report

Each stage consumes explicit configuration and upstream artifacts and emits a manifest linking its outputs to their provenance. Notebooks may illustrate results; production work must run through the package.
Hardware and execution requirements
- Start with the available M4 Mac, 16 GB unified memory.
- Validate FFT forward/backward operations, complex weights, physics gradients, and CPU/MPS numerical agreement before committing to MPS training. PyTorch exposes MPS as a supported device backend, but the project’s actual operations still require testing. PyTorch MPS documentation
- Provide the same configurations and resumable commands for a CUDA GPU with a target of at least 16 GB VRAM.
- Stream data from disk and process generation in bounded chunks; never load the complete dataset into memory.
- Reserve approximately 100 GB of disk space. Uncompressed reference fields occupy about 36 GB and derived model grids about 4 GB; remaining space covers environments, checkpoints, and artifacts.
- Benchmark representative generation, training, validation, and evaluation work before publishing runtime estimates. Extrapolate the full 50-run budget from measurements.
- If local execution is too slow or fails numerical compatibility checks, move the unchanged study to CUDA. Compare performance measurements on the same hardware and precision.
Required tests
- Solver accuracy, convergence, periodicity, conservation, and energy behavior.
- Dataset reproducibility, split isolation, checksums, chunked loading, and training-only normalization.
- Model shapes and gradients at all three resolutions; exact architectural identity between FNO variants.
- Physics derivatives on analytic functions; conservative derivative sums; zero-weight equivalence to supervised training.
- Gradient accumulation equivalence, checkpoint restoration, and interrupted-run recovery.
- Rollout feedback, periodic interpolation, metric formulas, failure handling, and report regeneration.
- A small integration fixture exercising the complete pipeline; fixture results are never used as research evidence.
Completion gates
Complete the work in this order:
1. Freeze the research protocol and verify numerical methods.
2. Generate and validate the complete dataset.
3. Verify architectures, losses, and hardware compatibility.
4. Complete the registered experiment matrix and locked evaluation.
5. Rebuild the research report from archived artifacts.
The project is complete when every reported result traces to a dataset version, configuration, checkpoint, and evaluation artifact, and another researcher can reproduce the workflow from the documentation.

