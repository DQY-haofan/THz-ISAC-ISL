"""
signal_model.py - Generate impaired signals per paper model
Implements Eq. (2) and expanded model from paper
"""

import numpy as np
from scipy.special import erfc
import utils

def generate_otfs_pilots(N_samples, M_frames, rng=None):
    """
    Generate OTFS pilot signals with Gaussian-like envelope
    Simplified from full OTFS for computational efficiency
    
    Args:
        N_samples: Number of samples per frame
        M_frames: Number of frames
        rng: Random number generator
        
    Returns:
        s: Complex baseband signal [M_frames x N_samples]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Generate complex Gaussian symbols (PAPR-clipped)
    s = rng.standard_normal((M_frames, N_samples)) + 1j * rng.standard_normal((M_frames, N_samples))
    s = s / np.sqrt(2)  # Unit power normalization
    
    # Apply PAPR clipping for PA efficiency (typical for satellite)
    papr_limit = 3.0  # 9.5 dB PAPR limit
    clip_level = np.sqrt(papr_limit)
    s_clipped = np.minimum(np.abs(s), clip_level) * np.exp(1j * np.angle(s))
    
    return s_clipped

def generate_phase_noise(N_samples, M_frames, Delta_nu, T_s, rng=None):
    """
    Generate Wiener process phase noise
    
    Args:
        N_samples: Samples per frame
        M_frames: Number of frames
        Delta_nu: 3-dB linewidth [Hz]
        T_s: Sample time [s]
        rng: Random number generator
        
    Returns:
        phi: Phase noise samples [M_frames x N_samples]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Wiener process increment standard deviation
    sigma_increment = np.sqrt(2 * np.pi * Delta_nu * T_s)
    
    # Generate phase noise for each frame
    phi = np.zeros((M_frames, N_samples))
    
    for m in range(M_frames):
        # Wiener process: phi[n] = phi[n-1] + increment
        increments = sigma_increment * rng.standard_normal(N_samples)
        phi[m, :] = np.cumsum(increments)
    
    # Check short-frame approximation validity
    frame_variance = 4 * np.pi * Delta_nu * utils.T_frame / 3
    actual_variance = np.var(phi[:, -1])
    
    if actual_variance > 0.1:
        print(f"Warning: Phase variance {actual_variance:.3f} > 0.1 rad², small-angle approx invalid")
    
    return phi

def apply_pa_nonlinearity(s_in, kappa, method='soft_limiter'):
    """
    Apply PA nonlinearity using Bussgang decomposition
    
    Args:
        s_in: Input signal
        kappa: Input back-off ratio
        method: 'soft_limiter' or 'saleh'
        
    Returns:
        s_out: PA output
        B: Bussgang gain
        eta: Distortion component
    """
    if method == 'soft_limiter':
        # Soft limiter model from Appendix C
        A_sat = 1.0
        
        # Apply nonlinearity element-wise
        s_out = np.zeros_like(s_in)
        mask = np.abs(s_in) <= A_sat
        s_out[mask] = s_in[mask]
        s_out[~mask] = A_sat * s_in[~mask] / np.abs(s_in[~mask])
        
        # Compute Bussgang parameters
        B, sigma_eta_sq = utils.compute_bussgang_gain(kappa)
        
        # Decompose: s_out = B * s_in + eta
        eta = s_out - B * s_in
        
    else:  # Saleh model
        # Saleh model parameters (typical for TWT)
        alpha_a, beta_a = 2.0, 1.0
        alpha_phi, beta_phi = 4.0, 9.0
        
        r = np.abs(s_in)
        theta = np.angle(s_in)
        
        # AM/AM conversion
        A_out = alpha_a * r / (1 + beta_a * r**2)
        
        # AM/PM conversion
        Phi_out = alpha_phi * r**2 / (1 + beta_phi * r**2)
        
        s_out = A_out * np.exp(1j * (theta + Phi_out))
        
        # Approximate Bussgang parameters
        B = np.mean(s_out * np.conj(s_in)) / np.mean(np.abs(s_in)**2)
        eta = s_out - B * s_in
    
    return s_out, B, eta

def compute_channel_response(distance, velocity, acceleration, f_c, t, f_offset=0):
    """
    Compute dynamic channel response H_dyn(t,f;η) from Eq. (5)
    现在包含加速度项（二次相位项）
    
    Args:
        distance: Range [m]
        velocity: Radial velocity [m/s]
        acceleration: Radial acceleration [m/s^2]
        f_c: Carrier frequency [Hz]
        t: Time samples [s]
        f_offset: Baseband frequency offset [Hz]
        
    Returns:
        H: Channel response
    """
    # Path loss amplitude
    beta_ch = utils.c / (4 * np.pi * distance * f_c)
    
    # Antenna gains (simplified, assuming perfect pointing)
    G_tx = utils.db_to_linear(50)  # 50 dBi typical for 0.5m at 300 GHz
    G_rx = G_tx
    beta_ch *= np.sqrt(G_tx * G_rx)
    
    # Carrier phase
    Phi_carrier = -2 * np.pi * f_c * distance / utils.c
    
    # Doppler shift and DSE - 现在包含加速度
    f_D = utils.compute_doppler_shift(velocity, f_c)
    
    # DSE phase from Eq. (7) - 包含速度和加速度的贡献
    # φ_D(t,f) = 2π(f/c)(v₀t + 0.5a₀t²)
    Phi_DSE = 2 * np.pi * (f_offset / utils.c) * (velocity * t + 0.5 * acceleration * t**2)
    
    # 载波频率的多普勒相位（主要贡献）
    Phi_doppler_carrier = 2 * np.pi * (f_c / utils.c) * (velocity * t + 0.5 * acceleration * t**2)
    
    # Combined channel response
    H = beta_ch * np.exp(1j * (Phi_carrier + Phi_doppler_carrier + Phi_DSE))
    
    return H

def add_dse_residual(signal, snr_linear, rng=None):
    """
    Add DSE compensation residual per Eq. (14)
    
    Args:
        signal: Input signal
        snr_linear: Linear SNR
        rng: Random number generator
        
    Returns:
        signal with DSE residual added
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # DSE residual variance (bounded by 0.001/SNR)
    sigma_dse_sq = 0.001 / snr_linear
    
    # Generate complex Gaussian residual
    e_dse = np.sqrt(sigma_dse_sq/2) * (rng.standard_normal(signal.shape) + 
                                       1j * rng.standard_normal(signal.shape))
    
    return signal + e_dse

def generate_impaired_signal(snr_db, M_frames=None, N_samples=None, rng=None):
    """
    Generate complete impaired signal per paper model
    
    Args:
        snr_db: SNR in dB
        M_frames: Number of frames (default from utils)
        N_samples: Samples per frame (default from utils)
        rng: Random number generator
        
    Returns:
        Dictionary with:
            - y: Received signal
            - s_clean: Clean transmitted signal
            - B: Bussgang gain
            - sigma_eff_sq: Effective noise variance
            - components: Dict of individual impairments
    """
    if M_frames is None:
        M_frames = utils.M_frames
    if N_samples is None:
        N_samples = utils.N_samples
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Time samples
    T_s = 1 / utils.B  # Sample time
    t = np.arange(N_samples) * T_s
    
    # Generate OTFS pilots
    s = generate_otfs_pilots(N_samples, M_frames, rng)
    
    # Generate phase noise
    phi = generate_phase_noise(N_samples, M_frames, utils.Delta_nu, T_s, rng)
    
    # Apply phase noise
    s_pn = s * np.exp(1j * phi)
    
    # Apply PA nonlinearity
    s_pa, B, eta = apply_pa_nonlinearity(s_pn, utils.kappa)
    
    # Compute channel response (包含加速度)
    H = compute_channel_response(utils.R_0, utils.v_rel, utils.a_rel, utils.f_c, t)
    
    # Apply channel
    y_channel = s_pa * H[np.newaxis, :]  # Broadcast over frames
    
    # Add DSE residual
    snr_linear = utils.db_to_linear(snr_db)
    y_dse = add_dse_residual(y_channel, snr_linear, rng)
    
    # Compute signal power
    P_signal = np.mean(np.abs(y_dse)**2)
    
    # Compute noise power from SNR
    N_0 = P_signal / snr_linear
    
    # Add AWGN
    n_awgn = np.sqrt(N_0/2) * (rng.standard_normal((M_frames, N_samples)) + 
                                1j * rng.standard_normal((M_frames, N_samples)))
    
    # Final received signal
    y = y_dse + n_awgn
    
    # Compute effective noise variance (Eq. 10)
    sigma_eta_sq = np.var(eta)
    Gamma_eff = utils.compute_effective_gamma(B, sigma_eta_sq)
    G_path = np.abs(H[0])**2  # Path gain
    
    sigma_eff_sq = N_0 + P_signal * G_path * Gamma_eff * np.exp(np.var(phi))
    
    return {
        'y': y,
        's_clean': s,
        'B': B,
        'sigma_eff_sq': sigma_eff_sq,
        'components': {
            'phase_noise': phi,
            'pa_distortion': eta,
            'channel': H,
            'awgn': n_awgn,
            'N_0': N_0,
            'P_signal': P_signal,
            'G_path': G_path,
            'Gamma_eff': Gamma_eff
        }
    }

def validate_signal_generation():
    """Run basic validation tests"""
    print("\n=== Signal Model Validation ===")
    
    # Test at 20 dB SNR
    snr_test = 20
    result = generate_impaired_signal(snr_test, M_frames=10)
    
    # Check power levels
    rx_power_dbw = utils.linear_to_db(np.mean(np.abs(result['y'])**2))
    clean_power_dbw = utils.linear_to_db(np.mean(np.abs(result['s_clean'])**2))
    
    print(f"Input SNR: {snr_test} dB")
    print(f"Clean signal power: {clean_power_dbw:.1f} dBW")
    print(f"Received signal power: {rx_power_dbw:.1f} dBW")
    print(f"Bussgang gain B: {result['B']:.3f}")
    print(f"Effective noise variance: {result['sigma_eff_sq']:.2e}")
    
    # Check phase noise statistics
    phase_var = np.var(result['components']['phase_noise'])
    expected_var = 4 * np.pi * utils.Delta_nu * utils.T_frame / 3
    print(f"\nPhase noise variance: {phase_var:.3f} rad² (expected: {expected_var:.3f})")
    
    # Verify path loss is reasonable
    path_loss_db = -utils.linear_to_db(result['components']['G_path'])
    print(f"Path loss: {path_loss_db:.1f} dB")
    
    if 180 < path_loss_db < 220:  # Typical range for LEO THz
        print("✓ Path loss in expected range")
    else:
        print("✗ Warning: Path loss outside typical range")
    
    print("=============================\n")