# GIFT algorithm

This is the algorithm reference reached from the architecture plate on the [project homepage](../README.md).
It describes what GIFT learns and what its components are. Measured numbers, data splits, training
budgets and run commands are kept in [EXPERIMENTS.md](../EXPERIMENTS.md) and [SETUP.md](SETUP.md);
this page contains no measurements of its own.

This page explains what GIFT learns and its components, reached from the architecture plate on the [project homepage](../README.md).
Measured values, data splits, training budgets and run commands are recorded in [EXPERIMENTS.md](../EXPERIMENTS.md) and [SETUP.md](SETUP.md);
this page contains no independent measurements.

## Overview

### What does GIFT learn?

Surrogates usually predict the field a fixed time lag ahead. GIFT changes the object of learning: it identifies the **continuous-time generator**, the dynamical operator that maps the current state field to its instantaneous time derivative,

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

where $\omega$ is the vorticity field of two-dimensional incompressible flow and $G^{\dagger}$ is the true system generator. The superscript distinguishes the true system generator from the approximate model obtained through learning; it is not an adjoint or a pseudoinverse. GIFT decomposes the generator model into a bias-field channel $C$, a linear channel $A(\omega)$ and a nonlinear channel $Q(\omega,\omega)$, each containing a learnable operator. The three channels differ in amplitude response — $A(\lambda\omega)=\lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$, with $\lambda$ a scalar amplitude-scaling factor — and field tomography separates and learns them from flow-field trajectories by exploiting that difference.

A state one finite lag later is merely the time integral of the generator, and the generator *is* the governing law. One mathematical object therefore serves both purposes at once. GIFT thereby transforms the surrogate model's prediction mechanism from a black-box mapping hidden in neural network weights into an explicitly parameterized continuous-time generator, which can also be mapped back to testable governing physical laws.

### Components

| Component | In one line |
| --- | --- |
| Generator model | A parameterized model learned from scalar-field data to approximate the continuous-time generator: a bias-field channel $C$, a linear channel $A(\omega)$ and a nonlinear channel $Q(\omega,\omega)$, each containing a learnable operator. |
| Field tomography | Separating and learning different components of the generator from flow-field trajectories by exploiting differences in amplitude responses among the bias-field, linear and nonlinear channels, which are trained **alternately**, with no homogeneity identity written down explicitly. |
| Quadratic field-interaction unit | Nonlinearity comes from a structured field operator rather than an activation function: two learnable spectral filters act on the same input field, their outputs multiply pointwise, and the units are summed with weights (panel E). |
| Learnable spectral filter | A linear operator that applies learnable weights to individual spatial wavenumber components in the Fourier domain. |
| High-frequency branch | The complementary band $Q_{21}\omega$ is predicted by a separately trained branch, which is then coupled with the generator in a single fourth-order Runge–Kutta integration (panel D). |
| Recursive local correction | A fixed, gradient-free rule applied after each complete Runge–Kutta step during recursive rollouts: local median residuals of the de-meaned field are clipped at an amplitude scale, under a safety cap on anomalous grid points (40, 90 and 160 for $N$ = 64, 96 and 128). |
| Physical interpretability | With the generator frozen, known Navier–Stokes equation terms are fitted to the output of each channel and the recovered equation parameters are compared with the true ones. |

### Architecture and prediction process

![GIFT architecture and prediction process](../assets/figures/gift_architecture.png)

**GIFT architecture and prediction process.** (A) Densely sampled continuous flow-field trajectories. (B) Training data are constructed through Fourier transformation and frequency band decomposition: the band-limited state enters the main path, the complementary band enters the high-frequency branch. (C) Field tomography exploits differences in amplitude responses to separate the bias-field, linear and nonlinear channels, which are built from quadratic field-interaction units. (D) The high-frequency branch. (E) GIFT produces nonlinear responses through two learnable spectral filters. (F) The GIFT prediction procedure advances the generator to the next instant by fourth-order Runge–Kutta integration.

### Where the exact definitions are recorded

- Frequency decomposition, the projections $P_{21}$/$Q_{21}$ and the coupled Runge–Kutta integration: [EXPERIMENTS.md](../EXPERIMENTS.md) sections 2.1 and 3.1.
- Recursive local correction thresholds and the anomalous-point cap: [EXPERIMENTS.md](../EXPERIMENTS.md) section 3.2.
- Evaluation metrics and the seed-averaging convention: [EXPERIMENTS.md](../EXPERIMENTS.md) section 3.4.
- Training budgets, data splits and model selection: [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md).
- How to read, recompute and verify the published results: [RESULTS.md](RESULTS.md).

## Detailed description

### What does GIFT learn?

Surrogates usually predict the field a fixed time lag ahead. GIFT changes the object of learning: it identifies the **continuous-time generator**, the dynamical operator that maps the current state field to its instantaneous time derivative,

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = G^{\dagger}(\omega),\qquad G(\omega) = C + A(\omega) + Q(\omega,\omega),$$

where $\omega$ is the vorticity field of two-dimensional incompressible flow and $G^{\dagger}$ is the true system generator. The superscript distinguishes the true generator from the learned approximate model and does not denote an adjoint or pseudoinverse. GIFT decomposes the generator model into a bias-field channel $C$, a linear channel $A(\omega)$ and a nonlinear channel $Q(\omega,\omega)$, each containing a learnable operator. Their amplitude responses differ, with $\lambda$ a scalar amplitude-scaling factor: $A(\lambda\omega)=\lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega)=\lambda^{2}Q(\omega,\omega)$. Field tomography exploits this difference to separate and learn the channels from flow-field trajectories.

A state one finite lag later is merely the time integral of the generator, and the generator *is* the governing law. One mathematical object therefore serves both purposes at once. GIFT thereby transforms the surrogate model's prediction mechanism from a black-box mapping hidden in neural network weights into an explicitly parameterized continuous-time generator, which can also be mapped back to testable governing physical laws.

### Components

| Component | In one line |
| --- | --- |
| Generator model | A parameterized model learned from scalar-field data to approximate the continuous-time generator: a bias-field channel $C$, a linear channel $A(\omega)$ and a nonlinear channel $Q(\omega,\omega)$, each containing a learnable operator. |
| Field tomography | Separating and learning different components of the generator from flow-field trajectories by exploiting differences in amplitude responses among the bias-field, linear and nonlinear channels, which are trained **alternately**, with no homogeneity identity written down explicitly. |
| Quadratic field-interaction unit | Nonlinearity comes from a structured field operator rather than an activation function: two learnable spectral filters act on the same input field, their outputs multiply pointwise, and the units are summed with weights (panel E). |
| Learnable spectral filter | A linear operator that applies learnable weights to individual spatial wavenumber components in the Fourier domain. |
| High-frequency branch | The complementary band $Q_{21}\omega$ is predicted by a separately trained branch, which is then coupled with the generator in a single fourth-order Runge–Kutta integration (panel D). |
| Recursive local correction | A fixed, gradient-free rule applied after each complete Runge–Kutta step during recursive rollouts: local median residuals of the de-meaned field are clipped at an amplitude scale, under a safety cap on anomalous grid points (40, 90 and 160 for $N$ = 64, 96 and 128). |
| Physical interpretability | With the generator frozen, known Navier–Stokes equation terms are fitted to the output of each channel and the recovered equation parameters are compared with the true ones. |

### Architecture and prediction process

![GIFT architecture and prediction process](../assets/figures/gift_architecture.png)

**GIFT architecture and prediction process.** (A) Densely sampled continuous flow-field trajectories. (B) Training data are constructed through Fourier transformation and frequency band decomposition: the band-limited state enters the main path, the complementary band enters the high-frequency branch. (C) Field tomography exploits differences in amplitude responses to separate the bias-field, linear and nonlinear channels, which are built from quadratic field-interaction units. (D) The high-frequency branch. (E) GIFT produces nonlinear responses through two learnable spectral filters. (F) The GIFT prediction procedure advances the generator to the next instant by fourth-order Runge–Kutta integration.

### Where the exact definitions are recorded

- Frequency decomposition, the projections $P_{21}$/$Q_{21}$ and the coupled Runge–Kutta integration: [EXPERIMENTS.md](../EXPERIMENTS.md) sections 2.1 and 3.1.
- Recursive local correction thresholds and the anomalous-point cap: [EXPERIMENTS.md](../EXPERIMENTS.md) section 3.2.
- Evaluation metrics and the seed-averaging convention: [EXPERIMENTS.md](../EXPERIMENTS.md) section 3.4.
- Training budgets, data splits and model selection: [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md).
- How to read, recompute and verify the published results: [RESULTS.md](RESULTS.md).
