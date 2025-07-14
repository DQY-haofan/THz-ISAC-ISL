# THz ISL ISAC Validation Project

## Overview

This project implements numerical simulations to validate the theoretical model presented in the paper "Fundamental Limits of THz Inter-Satellite ISAC Under Hardware Impairments". The code generates results for Section VIII (Numerical Results and Validation) of the paper.

## Updates (Latest Version)

### New Features:
1. **GMM Validation**: Detailed visualization of GMM approximation accuracy for PA distortion
2. **Acceleration Modeling**: Extended signal model to include quadratic phase terms for accelerated motion
3. **Sensitivity Analysis**: GMM component number (K) sensitivity study
4. **Enhanced Validation**: More comprehensive theoretical validation tests

## Project Structure

```
thz_isac_validation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── utils.py                       # Shared utilities and system parameters
├── signal_model.py                # Signal generation with hardware impairments
├── bcrlb_compute.py               # Bayesian Cramér-Rao Lower Bound computation
├── capacity_bounds.py             # Communication capacity analysis
├── plots.py                       # Visualization functions (enhanced)
├── main.py                        # Main simulation orchestrator
├── gmm_sensitivity_analysis.py    # GMM K-value sensitivity study (new)
├── technical_questions.md         # Technical discussion points
└── results/                       # Output directory (created automatically)
    ├── crlb_r_vs_snr.png         # Position CRLB plot
    ├── crlb_v_vs_snr.png         # Velocity CRLB plot
    ├── capacity_vs_snr.png        # Capacity comparison
    ├── capacity_sensitivity_*.png # Sensitivity analysis plots
    ├── gmm_validation.png         # Monte Carlo validation
    ├── gmm_validation_detailed.png # GMM fitting visualization (new)
    ├── crlb_vs_acceleration.png   # Acceleration impact (new)
    ├── gmm_k_sensitivity.png      # K-value analysis (new)
    ├── summary_figure.png         # 4-panel summary
    └── simulation_results.csv     # Numerical data
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Pandas
- scikit-learn (for Gaussian Mixture Model)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Main Simulation:
```bash
python main.py
```

### GMM Sensitivity Analysis (Optional):
```bash
python gmm_sensitivity_analysis.py
```

The simulation will:
- Perform unit checks and validation
- Run component tests
- Execute main simulations across SNR range (0-40 dB)
- Generate GMM validation plots
- Analyze acceleration impact
- Generate all plots and tables
- Save results in the `results/` directory

Expected runtime: 3-5 minutes (with 4-core parallelization)

## Key Features

### 1. Signal Model (`signal_model.py`)
- Generates OTFS pilot signals with PAPR clipping
- Implements Wiener process phase noise
- Models PA nonlinearity using Bussgang decomposition
- **NEW**: Includes acceleration terms (quadratic phase)
- Computes effective noise variance

### 2. BCRLB Computation (`bcrlb_compute.py`)
- Fits GMM to non-Gaussian PA distortion
- Implements Slepian-Bangs formula for Bayesian FIM
- **NEW**: Supports acceleration parameter estimation
- Computes parameter gradients for position/velocity/acceleration
- Includes Monte Carlo validation
- Joint penalty factor: exp(σ²_φ) × (1 + Γ_eff)

### 3. Capacity Analysis (`capacity_bounds.py`)
- Computes hardware-limited capacity upper bound
- Calculates capacity ceiling (saturation level)
- Performs sensitivity analysis for phase noise and distortion
- Compares with classical Shannon capacity

### 4. Visualization (`plots.py`)
- CRLB vs SNR plots (position and velocity)
- Capacity vs SNR with hardware limitations
- **NEW**: GMM fitting validation (histogram vs PDF)
- **NEW**: CRLB vs acceleration analysis
- Sensitivity analysis plots
- Summary multi-panel figure

## Key System Parameters

- Carrier frequency: 300 GHz (THz band)
- Bandwidth: 100 GHz
- LEO satellite range: 2000 km
- Relative velocity: 15 km/s
- **Relative acceleration: 5 m/s²** (typical LEO)
- Phase noise linewidth: 100 kHz
- PA input back-off: 3 dB (κ = 0.5)
- Frame duration: 100 μs

## Expected Results

1. **Sensing Performance (at 30 dB SNR)**:
   - Position BCRLB shows ~30× penalty vs AWGN
   - Sub-millimeter accuracy achievable
   - Velocity estimation in cm/s range
   - Acceleration impact manageable for LEO dynamics

2. **Communication Performance**:
   - Capacity saturates at ~4.3 bits/symbol
   - Significant gap from AWGN capacity at high SNR
   - Hardware impairments create performance ceiling

3. **Validation**:
   - GMM approximation with K=3 accurate within 10%
   - DSE residual contribution < 0.1%
   - High-SNR floors confirm theoretical predictions
   - Joint penalty factor correctly models combined impairments

## New Analyses

### GMM Component Selection
Run the sensitivity analysis to determine optimal K:
```bash
python gmm_sensitivity_analysis.py
```
Results show K=3 provides best balance of accuracy and efficiency.

### Acceleration Impact
The main simulation now includes acceleration sweep analysis, showing how relative acceleration affects estimation performance. Typical LEO accelerations (0-10 m/s²) result in manageable performance degradation.

## Notes

- Random seed set to 42 for reproducibility
- All physical units in SI (Hz, W, m, s, rad)
- Simplified ML estimators used for Monte Carlo validation
- Parallel computation using 4 CPU cores (configurable)

## Paper Reference

This code validates the theoretical framework presented in:
"Fundamental Limits of THz Inter-Satellite ISAC Under Hardware Impairments"
[Authors, Journal, Year]

## Technical Questions

See `technical_questions.md` for open questions requiring expert discussion.

## Contact

For questions about the implementation, please refer to the paper or contact the authors.