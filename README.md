# Calorimeter Energy Resolution Analysis

*A statistical study of detector response modelling using Maximum Likelihood Estimation, profile likelihoods, and resampling methods.*

**Iuliia Vitiugova** · Department of Applied Mathematics and Theoretical Physics  · University of Cambridge · 2025

---

## Overview

This repository presents a rigorous statistical analysis of **calorimeter energy resolution**, a central problem in experimental particle physics. The goal is to characterise the energy-dependent response of a particle detector — its bias, resolution, and noise contributions — using a hierarchy of statistical estimation approaches, from sample statistics through to a global simultaneous likelihood fit.

The study compares the statistical properties of several estimation frameworks and quantifies parameter uncertainties via both asymptotic (Hessian) errors and exact resampling methods.

---

## Problem Statement

Given a dataset of $(E_\text{true},\ E_\text{rec})$ pairs from a particle detector simulation, the measured reconstruction follows:

$$E_\text{rec} \mid E_0 \sim \mathcal{N}\!\left(\mu(E_0),\ \sigma(E_0)\right)$$

The response is modelled by two parametric functions:

**Mean (bias) model:**
$$\mu(E_0) = \lambda E_0 + \Delta$$

**Resolution model** (standard calorimeter parametrisation):
$$\frac{\sigma(E_0)}{E_0} = \sqrt{\frac{a^2}{E_0} + \frac{b^2}{E_0^2} + c^2}$$

| Parameter | Physical interpretation |
|-----------|------------------------|
| $\lambda$ | Energy scale (gain calibration) |
| $\Delta$  | Constant offset (pedestal / electronic noise) |
| $a$       | Stochastic (sampling) term $\propto 1/\sqrt{E}$ |
| $b$       | Noise term $\propto 1/E$ |
| $c$       | Constant systematic floor |

---

## Statistical Methods

### 1. Sample Estimators
Per-energy-slice sample mean and standard deviation, with analytical error propagation. Linear and resolution trends are extracted via **Least Squares** fitting using `iminuit`.

### 2. Individual Unbinned MLE
For each energy slice, an **unbinned Gaussian NLL** is minimised independently using `iminuit` (MIGRAD + HESSE). Parameters and uncertainties are compared against sample estimates.

### 3. Simultaneous Global Fit
A single **global NLL** is constructed over all events jointly, fitting $(\lambda, \Delta, a, b, c)$ simultaneously without intermediate aggregation. This is statistically more efficient and avoids information loss from per-slice compression.

### 4. Profile Likelihood & Confidence Contours
1D profile likelihoods $\Delta \text{NLL}(\theta_i)$ and 2D confidence contours (via `mnprofile` / `mncontour`) are computed for all five detector parameters, providing exact likelihood-based confidence intervals beyond Gaussian approximation.

### 5. Jackknife Resampling
Bias-corrected jackknife estimates of mean and standard deviation are computed per energy using the `resample` package, providing a non-parametric cross-check on uncertainties.

### 6. Non-Parametric Bootstrap Comparison
A full-analysis bootstrap ($n = 2500$ replicas) propagates statistical uncertainty end-to-end through all three estimation pipelines, enabling a direct comparison of their variance and consistency:
- Sample estimates + LS trend fits
- Per-energy MLE + trend fits
- Simultaneous fit

---

## Repository Structure

```
calorimeter/
├── notebooks/
│   └── solution.ipynb       # Full analysis with results and figures
├── s1_sol/                  # Analysis package
│   ├── data_loader.py       # Data I/O and grouping
│   ├── estimators.py        # Sample statistics
│   ├── mle_fits.py          # Per-energy unbinned Gaussian MLE
│   ├── fitting.py           # Trend fitting, bootstrap, jackknife
│   ├── simultaneous_fit.py  # Global simultaneous NLL fit
│   ├── profiling.py         # Profile likelihoods and 2D contours
│   └── plotting.py          # Publication-quality figures
├── sample.csv               # Dataset (E_true, E_rec)
├── results.json             # Best-fit parameters (auto-generated)
├── mphil.mplstyle           # Custom Matplotlib style
└── pyproject.toml           # Package configuration
```

---

## Installation

Requires **Python ≥ 3.9**.

```bash
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `iminuit`, `pandas`, `resample`, `jupyter`

---

## Usage

```bash
jupyter notebook notebooks/solution.ipynb
```

Results are saved to `results.json`; figures are written to `figs/`.

---

## Author

**Iuliia Vitiugova** — iv294@cam.ac.uk
