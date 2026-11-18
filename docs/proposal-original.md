
Introduction
Numerically simulating partial differential equations (PDEs) that govern physical systems—such as fluid flow, heat transfer, and wave propagation—is computationally expensive, particularly when repeated simulations across many initial conditions or parameter settings are required. Fourier Neural Operators (FNOs) are among the most prominent deep learning architectures for this task, performing convolution in the spectral (Fourier) domain to learn mappings between continuous function spaces rather than fixed-resolution grids. A parallel line of work, physics-informed learning, incorporates the governing PDE directly into the training objective as a soft regularization constraint, with the aim of improving physical consistency and out-of-distribution generalization beyond what pure data-fitting achieves. This project investigates, in a controlled setting, whether and how such physics constraints measurably improve a neural operator's rollout stability, physical conservation behavior, and generalization across spatial resolutions, using the one-dimensional viscous Burgers' equation as the benchmark system.   

Literature Survey
Li et al. [2] introduced the Fourier Neural Operator (FNO) for resolution-invariant function mappings, achieving strong speedups and accuracy on standard PDE benchmarks. Raissi, Perdikaris, and Karniadakis [3] introduced Physics-Informed Neural Networks (PINNs), incorporating PDE residuals directly into the neural network training loss. Li et al. [4] later developed the Physics-Informed Neural Operator (PINO), combining FNOs with physics-constrained losses to improve accuracy and super-resolution performance. However, recent work by Cardoso-Bihlo and Bihlo [5] emphasizes that standard residual-based soft penalties do not guarantee exact physical conservation. Furthermore, local spatial architectures such as U-Net [1] remain strong fixed-grid baselines when paired with standard interpolation. This motivates systematically testing whether soft physics constraints actually improve accuracy, long-horizon rollout stability, and physical conservation under strictly matched conditions.   
+ 4
Research Gaps
Physics-informed neural operator studies frequently report point-wise data-fit error (e.g., relative L 
2
​	
  error) but do not consistently separate local PDE-residual satisfaction from global, physically meaningful conservation error (such as total mass/momentum and energy decay), leaving it unclear whether a low residual translates into physically consistent multi-step predictions.   
+ 4
FNO's zero-shot resolution-generalization property is often stated as an architectural given rather than benchmarked against a spatial convolutional baseline adapted with standard grid interpolation, making claimed spectral advantages difficult to isolate.   

The distinction between continuous automatic-differentiation residuals (used in coordinate-based PINNs) and discrete grid-based residual computation (finite-difference or spectral derivatives necessary for grid-based operators like FNO) is rarely evaluated in terms of its impact on conservation drift during autoregressive rollouts.   

Proposed Solution
The project follows a controlled 4-stage pipeline:   

Data Generation: A numerical solver (pseudo-spectral / finite-difference with Runge-Kutta time integration) is implemented to generate exact ground-truth trajectories of the 1D periodic viscous Burgers' equation across varied initial conditions, forming a clean, self-contained dataset with no external data dependency.   

Spatial Baseline Model (1D U-Net): A standard 1D convolutional U-Net is trained for next-step prediction at a fixed base resolution (N=256) under a purely supervised Mean Squared Error (MSE) data loss. For higher-resolution testing, it is paired with cubic/linear grid interpolation.   

Unconstrained Neural Operator (Vanilla FNO): A standard 1D FNO (utilizing spectral convolution layers with truncated Fourier modes) is trained under the exact same supervised MSE loss and training dataset.   
Physics-Constrained FNO (PINO-style): The identical FNO architecture is retrained with a composite loss combining supervised data MSE and a discrete PDE-residual penalty (L 
total
​	
 =L 
data
​	
 +λL 
PDE
​	
 ), where temporal and spatial derivatives in the residual are computed on the predicted grid in conservative flux form.   
DOC-20260831-WA0002.pdf
All three trained models—1D U-Net, vanilla FNO, and physics-constrained FNO—are evaluated on a shared diagnostic suite:   

Autoregressive Rollout Error: Each model's one-step prediction is iteratively fed back as input over multiple time steps, tracking relative L 
2
​ error accumulation over the rollout horizon.   

Mass Conservation Drift and Energy Decay vs. PDE Residual: Since total mass/momentum ∫u(x,t)dx is strictly conserved for the periodic Burgers' equation regardless of viscosity, drift in this integral (alongside physical kinetic energy dissipation) is tracked over the rollout independently from the local PDE-residual value. This directly separates "satisfies the local equation" from "conserves the global physical quantity".   
Zero-Shot Resolution Generalization: Models are trained exclusively at base spatial resolution (N=256) and evaluated zero-shot at higher spatial resolutions (N=512 and N=1024) without retraining. 

Expected Outcome
A quantified, empirically grounded evaluation of whether adding a soft physics-residual constraint measurably reduces conservation drift, improves zero-shot super-resolution accuracy, and stabilizes multi-step autoregressive rollouts relative to an unconstrained FNO and a spatial convolutional baseline—including an honest accounting of computational trade-offs and sensitivity to the physics-loss weight λ.   

Technologies Used
Python, PyTorch, NumPy, SciPy (numerical solver implementation), Matplotlib (diagnostic visualization)   

References
[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in Proc. MICCAI, 2015, pp. 234–241.   
DOC-20260831-WA0002.pdf
[2] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," in Proc. ICLR, 2021 (arXiv:2010.08895).   
DOC-20260831-WA0002.pdf
[3] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," Journal of Computational Physics, vol. 378, pp. 686–707, 2019.
[4] Z. Li et al., "Physics-Informed Neural Operator for Learning Partial Differential Equations," ACM / IMS Journal of Data Science, vol. 1, no. 3, pp. 1–27, 2024 (arXiv:2111.03794).
[5] A. Cardoso-Bihlo and R. O. Bihlo, "Physics-informed neural networks for the conservative solution of partial differential equations," Journal of Computational Physics / Numerical Analysis Preprints, 2022.