
# Gradient-Free Ensemble Transform Langevin Dynamics with MMD

This repository contains implementations of **gradient-free ensemble transform Langevin dynamics (GF-ETLD)** for likelihood-free Bayesian inference using **maximum mean discrepancy (MMD)** as the data-fitting term.

The code demonstrates robust Bayesian inference through generalized posteriors that can handle model misspecification, with applications to location estimation and chaotic dynamical systems.

## Overview

This repository implements and evaluates GF-ETLD for likelihood-free inference with three key experimental scenarios:

**Well-specified models:**
- Stochastic Lorenz96 dynamical system 

**Misspecified models:**
- Gaussian location model with contamination
- Uniform location model with contamination

**Key innovation:**
- Affine-invariant preconditioning for enhanced sampling efficiency
- Statistical linearization for gradient-free parameter updates
- MMD-based generalized Bayesian posteriors for robust inference

## Generalized Bayesian posterior

We target the generalized posterior

$$
\pi(\theta \mid Y)
\propto
\pi_{0}(\theta)\,
\exp(-\beta\,\mathrm{MMD}^2(P_\theta, P_{\text{data}})),
$$

where

* $\pi_{0}(\theta)$ is the prior density  
* $\beta$ is the temperature parameter  
* $\mathrm{MMD}^2(P_\theta,P_{\text{data}})$ measures the discrepancy between model and data distributions  

## GF-ETLD Algorithm

The method evolves an ensemble of particles $\theta^m_s, m=1,\dots,M$ for time $s>0$ using an **affine-invariant** preconditioner $C_s$ (empirical covariance) and **statistical linearization**.  
The key surrogate for the Jacobian in the gradient of the MMD is

$$
C^{\theta x^j}_s
\approx
C_s \nabla_{\theta^m} G_{\theta_s^m}(u^j).
$$

This leads to the interacting particle system

$$ d\theta_s^{m}
= -\bigl(C_s \nabla_{\theta^m} \log \pi_{\text{prior}}(\theta^m) - \beta g_s^{mj}\bigr) ds
+ \frac{D+1}{M}\bigl(\theta_s^m-\bar{\theta}_s\bigr) ds
+ \sqrt{2}\, C_s^{1/2} dW_s^{m}, $$

for $m=1,\dots,M$, where $W_s^{m}$ is a $D$-dimensional standard Brownian motion and

$$ g_s^{mj}
= \frac{2}{J(J-1)}
  \sum_{l=1,\,l\neq j}^{J} C_s^{\theta x^j}
     \nabla_{x^{mj}} k(x^{mj}_s, x^{ml}_s)
  - \frac{2}{JN}
  \sum_{j=1}^{J}\sum_{n=1}^{N} C_s^{\theta x^j}
     \nabla_{x^{mj}} k(x^{mj}_s, y^{n}). $$


## Project structure

### Core implementation (`gfetld_mmd/`)

#### `gfetld.py`
**GF-ETLD sampler implementation**

Core sampler with affine-invariant preconditioning, correction and noise terms.

**Key Features:**
- Ensemble-based parameter evolution
- Affine-invariant covariance adaptation
- Statistical linearization for gradient approximation
- Configurable hyperparameters through `GFETLDConfig`

#### `mmd.py`
**Maximum Mean Discrepancy Computation**

Implements unbiased MMD² estimation and gradient computation.

**Key Features:**
- Unbiased MMD² estimator for sample sizes ≥ 2
- Stacked g-vector computation for ensemble updates
- Efficient kernel-based distance computation

#### `kernels.py`
**Kernel functions**

Gaussian kernel implementation with gradient computation w.r.t. outputs.

**Key features:**
- Gaussian kernel with configurable bandwidth
- Gradient computation for MMD optimization
- Extensible design for alternative kernel functions

#### `prior.py`
**Prior distribution utilities**

Prior distribution utilities for Bayesian inference.

**Key Features:**
- Prior log-density and gradient computation
- Configurable prior parameters
- Support for diagonal covariance structures

#### `simulators.py`
**Forward model implementations**

Collection of simulator functions for different experimental scenarios.

**Key features:**
- Gaussian and uniform location generators
- Stochastic Lorenz96 with AR(1) residual forcing
- RK4 numerical integration for dynamical systems

#### `plots.py`
**Visualization utilities**

Matplotlib helpers for generating paper-style figures and analysis plots.

**Key Features:**
- Contamination curve plotting
- RMSE visualization
- Histogram generation for parameter distributions

### Experiments (`experiments/`)

#### `exp_gaussian_location.py`
**Gaussian location model experiment**

Tests robustness under Gaussian location model with outlier contamination.

**Key features:**
- Contamination levels: 0%, 10%, 20%
- RMSE vs contamination analysis
- Posterior mean estimation 

#### `exp_uniform_location.py` 
**Uniform location model experiment**

Evaluates performance under uniform location model misspecification.

**Key features:**
- Uniform data generation with contamination
- Robustness comparison with Gaussian model
- Posterior uncertainty quantification

#### `exp_stoch_lorenz96.py`
**Stochastic Lorenz96 experiment**

Well-specified model test using a parametrized version of the Lorenz96 model.

**Key features:**
- Lorenz96 system with K=8, F=10
- RK4 integration with dt=3/40, T=2.5
- AR(1) residual forcing for stochasticity
- Parameter estimation 

## Dependencies

```python
numpy            # Numerical computations and array operations
matplotlib       # Plotting and visualization
```

## Getting started

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate   # (optional)
pip install numpy matplotlib
```

### 2. Run experiments

From the folder containing `experiments/`:

```bash
# Gaussian location model with contamination
python -m experiments.exp_gaussian_location

# Uniform location model with contamination  
python -m experiments.exp_uniform_location

# Stochastic Lorenz96 dynamical system
python -m experiments.exp_stoch_lorenz96
```

### 3. Configuration parameters

Key hyperparameters in `GFETLDConfig`:

```python
step_size: float = 1e-3          # Euler-Maruyama step size
n_steps: int = 200               # Number of sampling steps
ensemble_size: int = 100         # Number of particles (M)
n_sim_per_particle: int = 20     # Simulator replicates per particle (J)
kernel_bandwidth: float = 1.0    # Gaussian kernel bandwidth
temperature: float = 1.0         # Beta parameter for data-fitting term
```

**Tuning guidelines:**
- Increase `n_sim_per_particle` to reduce variance in MMD estimates
- Adjust `kernel_bandwidth` to match data scale (higher for high-dimensional outputs)
- Tune `temperature` to control data-fitting vs prior balance
- Use more `n_steps` for tighter convergence

## Output structure

Each experiment generates results in `experiments/outputs/`:

**Gaussian/Uniform location models:**
- `rmse_vs_contamination.png`: Visualization of robustness
- `summary.csv`: Contamination levels, posterior means, RMSE values

**Lorenz96 experiment:**
- `summary.csv`: Posterior means and absolute errors for each parameter

## Key concepts

### Likelihood-free inference
- Simulator-based models without explicit likelihood functions
- MMD-based discrepancy measures for data-fitting
- Robust alternatives to traditional likelihood methods

### Statistical linearization
- Gradient-free approximation of simulator sensitivities
- Cross-covariance based parameter-output relationships
- Efficient computation without autodifferentiation

## Research applications

This implementation addresses:

- **Robust Bayesian inference:** Handling model misspecification through generalized posteriors
- **Complex simulator models:** Efficient sampling for expensive forward models
- **Parameter inference:** Accurate inference in dynamical systems and location models
- **Uncertainty quantification:** Ensemble-based posterior characterization

## Extensions

The framework supports easy extensions:

- **Alternative kernels:** Implement new kernel functions in `kernels.py`
- **Different priors:** Add prior distributions with `logpdf_grad` interface
- **New simulators:** Extend `simulators.py` with additional forward models

## Usage Notes

- Gaussian/Uniform experiments include contamination scenarios for robustness testing
- Lorenz96 uses RK4 integration with default parameters (K=8, F=10, dt=3/40, T=2.5)
- Unbiased MMD² estimator requires both sample sizes ≥ 2
- Monitor ensemble spread and convergence during sampling
- Adjust kernel bandwidth based on output dimensionality and scale

For detailed algorithmic descriptions and theoretical foundations, refer to the individual module documentation and associated research papers.
