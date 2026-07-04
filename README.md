# adjoint-based sampling method

This repository contains PyTorch experiments for **adjoint-based sampling methods** based on the work *Analyzing Training Dynamics of Transformers through Data-Driven Nonlocal Mean Field Control*. The goal is to study how the backward adjoint information can be used to guide sampling dynamics and to provide clean, reproducible experiments on low-dimensional target distributions.

The project is motivated by the connection between generative modeling, controlled particle dynamics, and adjoint-based optimality conditions. In particular, it focuses on learning transport dynamics that move samples from a simple initial distribution toward a target distribution while analyzing the corresponding forward trajectories and backward adjoint variables.

## Overview

Modern generative modeling often requires sampling from complex target distributions. This project explores a control-theoretic perspective: instead of directly prescribing a fixed sampler, we learn a dynamical system that transports particles from an initial distribution to a target distribution.

The method combines three ideas:

- **Particle dynamics:** samples evolve through a learned velocity field.
- **Adjoint equations:** backward variables quantify sensitivity of the objective with respect to particle trajectories.
- **Distribution matching:** the terminal particle distribution is trained to match a target distribution.

This repository is intended as an educational and experimental implementation for understanding adjoint-based sampling and its connection to generative modeling.

## Features

- PyTorch implementation of particle-based sampling dynamics
- Low-dimensional synthetic target distributions
- Forward particle trajectory visualization
- Backward adjoint trajectory visualization
- Experiments connecting sampling, optimal control, and generative modeling
- Clean code structure suitable for further extensions

## Mathematical Background

We consider particles evolving according to a learned dynamics

\[
\frac{dX_t}{dt} = v_\theta(t, X_t),
\]

where \(X_t\) denotes the particle state and \(v_\theta\) is a trainable velocity field. Starting from an initial distribution \(\mu_0\), the goal is to transport samples so that the terminal distribution approximates a target distribution \(\mu_1\).

A typical objective has the form

\[
\mathcal{L}(\theta)
=
\text{TerminalLoss}\big((X_T)_\# \mu_0, \mu_1\big)
+
\lambda \int_0^T \|v_\theta(t, X_t)\|^2 \, dt.
\]

The corresponding adjoint variables evolve backward in time and provide sensitivity information for the learned trajectories. This gives a useful way to analyze which particles or regions of the distribution contribute most strongly to the terminal objective.

## Repository Structure

```text
adjoint-based-sampling/
├── README.md
├── requirements.txt
├── src/
│   ├── particle/
│   ├── sc_new/
│   ├── sc_new_V_attetion/
│   └── utils/
├── experiments/
├── figures/
└── results/
```

Suggested organization:

- `src/particle/`: self-attention particle dynamics
- `src/sc_new/`: soft-constraint mfc formulation
- `src/sc_new_V_attention/`: soft-constraint mfc formulation with trainable matrix V only
- `src/utils/`: plotting, data generation, and helper functions
- `experiments/`: runnable training scripts
- `figures/`: generated visualizations
- `results/`: saved experiment outputs

## Installation

Clone the repository:

```bash
git clone https://github.com/dixiwang819/adjoint-based-sampling.git
cd adjoint-based-sampling
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet available, the core dependencies are:

```bash
pip install numpy scipy matplotlib torch scikit-learn tqdm
```

## Quick Start

Run a toy 2D sampling experiment:

```bash
python experiments/train_toy_2d.py
```

Generate visualizations:

```bash
python experiments/plot_results.py
```

Expected outputs include:

- initial particle distribution
- learned terminal distribution
- target distribution
- particle trajectories
- adjoint trajectory visualization

## Example Experiments

### Gaussian to Two-Moon Distribution

Particles are initialized from a Gaussian distribution and transported toward a two-moon target distribution.

This experiment illustrates how the learned dynamics bends and separates particle trajectories to match a non-Gaussian target.

### Gaussian to Checkerboard Distribution

Particles are transported from a Gaussian distribution toward a checkerboard-type target distribution.

This experiment tests whether the learned sampler can capture multi-modal geometric structure.

### Adjoint-Based Sensitivity Visualization

After training, backward adjoint variables are computed along particle trajectories. Large adjoint magnitudes indicate particles or regions that have stronger influence on the terminal objective.

## Example Results

Add figures here after running experiments:

```markdown
| Initial Samples | Learned Samples | Target Distribution |
|---|---|---|
| ![](figures/initial.png) | ![](figures/learned.png) | ![](figures/target.png) |
```

For trajectory visualization:

```markdown
| Forward Particle Trajectories | Backward Adjoint Trajectories |
|---|---|
| ![](figures/forward_trajectories.png) | ![](figures/adjoint_trajectories.png) |
```

## Why This Project?

This project is designed to connect several themes in modern machine learning and applied mathematics:

- generative modeling
- diffusion-inspired sampling
- optimal control
- adjoint sensitivity analysis
- particle methods
- distribution-to-distribution transport

The goal is not only to train a sampler, but also to understand the learned dynamics through the associated adjoint system.

## Current Status

This repository is under active development. Planned improvements include:

- cleaner experiment configuration files
- additional target distributions
- comparison with baseline flow-based samplers
- improved visualization of adjoint magnitudes
- GPU support for larger particle systems
- documentation of the mathematical derivation

## Disclaimer

This is an independent educational and experimental implementation. It is not the official implementation of any paper.

## License

This project is released under the MIT License.
