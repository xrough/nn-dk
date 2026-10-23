# Numerical and statistical conventions

## Observation and initialization

Let (H=2\pi/M) and ψ_j(x)=B_3((x-jH)/H), periodically summed. The centered cardinal cubic spline is nonnegative, has support [-2,2], and its translates form a partition of unity. Store masses (R_j=N^{-1}\sum_iψ_j(X_i)), not densities. Derivatives include the appropriate factors (H^{-1}) and (H^{-2}).

Finite-volume cell i is [ih,(i+1)h), with its density constant on that interval. The observation matrix is (A_{ji}=h^{-1}\int_{ih}^{(i+1)h}ψ_j(x)dx). Quadrature splits at every spline breakpoint and is polynomial-exact. Initial field probabilities are integrated from the normalized profile. DK starts from the histogram of an IID particle sample. Its initial observed path differs from the exact smooth particle observation by a documented reconstruction error.

For uniform particles, the exact observed quadratic variation before division by N is (C_{j\ell}=2D/(2\pi)\intψ'_jψ'_\ell dx). The heat oracle Fourier-transforms the actual combined test functions. This retains filtering and aliasing effects.

## Finite-volume comparators

The incidence matrix maps each oriented face current to opposite changes in its adjacent cells. It contains no hidden h factor. Deterministic flux is (-D(ρ_{i+1}-ρ_i)/h-ρ_{face}(V'*ρ)_{face}). The interaction force integrates each Fourier mode against the piecewise constant reconstruction, including its cell sinc factor.

One face Gaussian is shared with opposite signs between adjacent masses. Its flux amplitude is sqrt(2D max(ρ_face,0)/(Nh)). Explicit stepping obeys the checked diffusion/advection CFL and centered-advection Peclet conditions. The zero square-root floor is part of the declared discrete model. Stopping a negative path is a validity decision, not a positivity-preserving numerical correction.

Linear fluctuations start with scaled covariance diag(p)-p p^T for IID cell probabilities. Their covariance update is (T C T^T+dt Q), where T=I+dt J is the drift linearization. Previously saved cross covariances advance with T^T. This produces one joint covariance over every queried time and mode.

## Learning and uncertainty

Project A is an ensemble-conditioned finite-lag Gaussian kernel. Its explicit N^-1/2 scaling is a model assumption tested at fixed N first. Its outputs do not identify instantaneous SDE coefficients or establish a Markov property of the observed particles.

Project B learns means and N-scaled covariances. Distinct disjoint pairs a,b contribute N/2 (Y_a-Y_b)(Y_a-Y_b)^T. All possible pairs are never counted as independent labels. Its Cholesky head ensures one positive semidefinite multitime covariance. Direct regression baselines report any covariance inconsistency without post-test projection.

The evaluator bootstraps whole independent paths within each context. Reference and simulator sampling intervals are reported separately. Optimizer seeds are separate fits; they do not supply uncertainty about resampling the training dataset. No test mean is used for centering or normalization. Analytic uniform-observation fluctuation units provide the default error scales.

The draft's 5% tolerance is a research objective. With R independent Gaussian observations, a variance estimate's approximate relative 95% interval half-width is 1.96 sqrt(2/(R-1)). Small smoke ensembles cannot resolve that tolerance. Calibration therefore separates implementation checks from unresolved scientific certification. Spatial and initial-projection errors remain separate from neural error.
