# Quantum Trajectory Reconstruction (QTR)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Quantum OS Powered by © Stillmind Hub.**

## Overview

This repository provides a research-grade Python prototype for reconstructing quantum trajectories from weak measurements using Bayesian inference (Particle Filter) and Deep Learning (GRU).

The project explores the intersection of quantum mechanics and probabilistic inference, providing a robust pipeline to simulate 1D wavefunctions and reconstruct their most probable paths under measurement constraints.

## Features

- **High-Fidelity Physics**: 1D Schrödinger equation solver using the Split-Operator FFT method.
- **Weak Measurement Simulation**: Gaussian noise modeling for continuous position monitoring.
- **Advanced Inference Engine**:
  - **Particle Filter**: Sequential Monte Carlo for real-time Bayesian estimation.
  - **Neural Inference**: Recurrent Neural Network (GRU) for pattern-based trajectory reconstruction.
- **Comprehensive Visualization**: Automated plotting of wavefunction evolution and trajectory comparisons.

## Installation

```bash
pip install numpy scipy matplotlib pandas torch
```

## Usage

Run the main simulation with default parameters:

```bash
python main.py --potential harmonic --steps 500 --neural
```

### CLI Arguments

- `--potential`: Type of potential (`free` or `harmonic`).
- `--steps`: Number of simulation time steps.
- `--noise`: Standard deviation of the measurement noise.
- `--particles`: Number of particles for the filter.
- `--neural`: Enable the neural network inference module.

## Project Structure

```text
├── core/
│   ├── schrodinger_solver.py  # TDSE Solver
│   └── potentials.py          # Potential definitions
├── measurement/
│   └── weak_measurement.py    # Weak measurement logic
├── inference/
│   ├── particle_filter.py     # Bayesian inference
│   └── neural_inference.py    # Deep learning inference
├── simulation/
│   └── experiment.py          # Orchestration
├── visualization/
│   └── plotter.py             # Data visualization
├── data/                      # Saved results
└── main.py                    # Entry point
```

## Scientific Goal

This prototype aims to demonstrate that quantum trajectory reconstruction can be framed as a classical probabilistic inference problem. By combining traditional Bayesian methods with modern AI, we can achieve high-precision state estimation even under significant measurement noise.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

---
**Quantum OS Powered by © Stillmind Hub.**
