"""
utils.py - Shared utilities and constants for THz ISL ISAC validation
Maps to paper parameters and provides common functions
"""

import numpy as np

# Physical constants (SI units)
c = 3e8  # Speed of light [m/s]

# System parameters from paper
f_c = 300e9  # Carrier frequency [Hz] - 300 GHz
B = 100e9    # Bandwidth [Hz] - 100 GHz
v_rel = 15e3 # Relative velocity [m/s] - 15 km/s
a_rel = 5.0  # Relative acceleration [m/s^2] - typical for LEO orbital dynamics
R_0 = 2000e3 # Nominal range [m] - 2000 km
Delta_nu = 100e3  # Phase noise linewidth [Hz] - 100 kHz
T_frame = 100e-6  # Frame duration [s] - 100 μs

# PA parameters (Appendix C)
kappa = 0.5  # Input back-off ratio (3 dB IBO)
A_sat = 1.0  # PA saturation level (normalized)

# Hardware impairments
# IQI已通过数字预失真补偿到-40dBc以下，因此忽略
# IRR = 30  # Image rejection ratio [linear] - REMOVED
sigma_t_ADC = 1e-12  # ADC clock jitter [s]

# OTFS parameters
N_d = 256  # Number of Doppler bins
N_tau = 64  # Number of delay bins

# Simulation parameters
N_samples = 1024  # Samples per frame
M_frames = 100    # Number of frames for averaging
N_monte_carlo = 1000  # Monte Carlo trials

# Antenna parameters
D_ant = 0.5  # Antenna diameter [m]
theta_3dB = 1.02 * (c/f_c) / D_ant  # 3dB beamwidth [rad]
g_ant = 2.77 / (theta_3dB**2)  # Antenna gain rolloff factor [rad^-2]

# Set global random seed for reproducibility
np.random.seed(42)

def db_to_linear(db_value):
    """Convert dB to linear scale"""
    return 10**(db_value/10)

def linear_to_db(linear_value):
    """Convert linear to dB scale"""
    return 10 * np.log10(np.maximum(linear_value, 1e-30))

def snr_db_to_noise_power(snr_db, signal_power=1.0):
    """Convert SNR in dB to noise power N0"""
    snr_linear = db_to_linear(snr_db)
    return signal_power / snr_linear

def create_rng(seed):
    """Create a new random number generator with given seed"""
    return np.random.default_rng(seed)

def compute_path_loss(distance, frequency):
    """
    Compute free-space path loss
    Args:
        distance: Distance in meters
        frequency: Frequency in Hz
    Returns:
        Path loss (linear scale, < 1)
    """
    wavelength = c / frequency
    return (wavelength / (4 * np.pi * distance))**2

def compute_doppler_shift(velocity, frequency):
    """
    Compute Doppler shift
    Args:
        velocity: Relative radial velocity [m/s]
        frequency: Carrier frequency [Hz]
    Returns:
        Doppler shift [Hz]
    """
    return -frequency * velocity / c

def compute_dse_bandwidth():
    """Compute differential Doppler across bandwidth"""
    return (v_rel / c) * B  # ~5 MHz for our parameters

def regularize_matrix(matrix, epsilon=1e-12):
    """
    Regularize matrix to ensure positive definiteness
    Args:
        matrix: Input matrix
        epsilon: Regularization parameter
    Returns:
        Regularized matrix
    """
    # Add small diagonal loading
    reg_matrix = matrix + epsilon * np.eye(matrix.shape[0])
    
    # Check condition number
    cond = np.linalg.cond(reg_matrix)
    if cond > 1e12:
        print(f"Warning: Matrix condition number {cond:.2e} > 1e12")
    
    return reg_matrix

def assert_units():
    """Verify physical parameter ranges"""
    assert 0 < B < 1e12, f"Bandwidth {B/1e9:.1f} GHz out of range"
    assert 0 < f_c < 1e12, f"Carrier frequency {f_c/1e9:.1f} GHz out of range"
    assert 0 < v_rel < 30e3, f"Relative velocity {v_rel/1e3:.1f} km/s unrealistic"
    assert 0 < kappa <= 0.5, f"PA back-off κ={kappa} invalid (must be ≤0.5)"
    assert 0 < Delta_nu < 1e6, f"Phase noise linewidth {Delta_nu/1e3:.1f} kHz out of range"
    print("Unit checks passed")

def compute_bussgang_gain(kappa_val):
    """
    Compute Bussgang gain for clipped Gaussian input (Appendix C)
    Args:
        kappa_val: Input back-off ratio P_in/A_sat^2
    Returns:
        B: Bussgang gain
        sigma_eta_sq: Distortion power (normalized)
    """
    assert kappa_val <= 0.5, f"κ={kappa_val} too large for accurate Bussgang"
    
    # Exact formula from Appendix C
    from scipy.special import erfc
    
    B = 1 - np.exp(-1/kappa_val) - np.sqrt(np.pi/(2*kappa_val)) * erfc(1/np.sqrt(2*kappa_val))
    
    # Distortion power
    sigma_eta_sq = 1 - 2*np.exp(-1/kappa_val) + np.exp(-2/kappa_val) - B**2
    
    # Taylor approximation for verification
    B_taylor = 1 - 1.5*kappa_val + 15/8*kappa_val**2
    
    if np.abs(B - B_taylor) > 0.1:
        print(f"Warning: Bussgang gain {B:.3f} differs from Taylor {B_taylor:.3f}")
    
    return B, sigma_eta_sq

def compute_effective_gamma(B, sigma_eta_sq, P_in=1.0):
    """
    Compute effective distortion factor Γ_eff (Eq. 66 in paper)
    简化版：IQI已补偿，只考虑PA和ADC
    
    Args:
        B: Bussgang gain
        sigma_eta_sq: PA distortion power
        P_in: Input power
    Returns:
        Gamma_eff: Effective distortion factor
    """
    # Components from Eq. 66
    pa_term = sigma_eta_sq / (B**2 * P_in)
    # IQI term removed - compensated to below -40dBc
    adc_term = (2 * np.pi * sigma_t_ADC * B/np.sqrt(12))**2
    
    Gamma_eff = pa_term + adc_term
    return Gamma_eff

def save_results(data, filename, directory='results'):
    """Save data to CSV file"""
    import os
    import pandas as pd
    
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    if isinstance(data, dict):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    else:
        np.savetxt(filepath, data, delimiter=',')
    
    print(f"Saved: {filepath}")

def print_system_summary():
    """Print key system parameters for verification"""
    print("\n=== THz ISL ISAC System Parameters ===")
    print(f"Carrier frequency: {f_c/1e9:.0f} GHz")
    print(f"Bandwidth: {B/1e9:.0f} GHz")
    print(f"Relative velocity: {v_rel/1e3:.0f} km/s")
    print(f"Nominal range: {R_0/1e3:.0f} km")
    print(f"Phase noise linewidth: {Delta_nu/1e3:.0f} kHz")
    print(f"Frame duration: {T_frame*1e6:.0f} μs")
    print(f"PA back-off (IBO): {linear_to_db(1/kappa):.1f} dB")
    print(f"DSE bandwidth: {compute_dse_bandwidth()/1e6:.1f} MHz")
    
    # Compute key derived quantities
    path_loss_db = linear_to_db(compute_path_loss(R_0, f_c))
    print(f"\nFree-space path loss: {-path_loss_db:.1f} dB")
    
    B_gain, sigma_eta_sq = compute_bussgang_gain(kappa)
    print(f"Bussgang gain B: {B_gain:.3f}")
    print(f"PA distortion σ²_η: {sigma_eta_sq:.3f}")
    
    Gamma_eff = compute_effective_gamma(B_gain, sigma_eta_sq)
    print(f"Effective distortion Γ_eff: {Gamma_eff:.3f}")
    
    # Capacity ceiling
    sigma_phi_sq = 4 * np.pi * Delta_nu * T_frame / 3  # Short-frame approx
    C_sat = 0.5 * np.log2(1 + np.exp(-sigma_phi_sq) / Gamma_eff)
    print(f"\nPhase noise variance: {sigma_phi_sq:.3f} rad²")
    print(f"Capacity ceiling: {C_sat:.2f} bits/symbol")
    print("=====================================\n")