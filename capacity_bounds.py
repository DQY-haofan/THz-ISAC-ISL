"""
capacity_bounds.py - Calculate capacity upper bounds and sensitivities
Implements Eq. (65) and capacity ceiling analysis from paper
"""

import numpy as np
import utils

def compute_capacity_upper_bound(snr_db, sigma_phi_sq=None, Gamma_eff=None):
    """
    Compute capacity upper bound from Eq. (65)
    C ≤ (1/2) log₂(1 + ρ·e^(-σ²_φ)/(1 + Γ_eff))
    
    Args:
        snr_db: SNR in dB
        sigma_phi_sq: Phase noise variance (default: computed from system params)
        Gamma_eff: Effective distortion factor (default: computed from system params)
        
    Returns:
        C: Capacity upper bound [bits/symbol]
    """
    # Convert SNR to linear
    rho = utils.db_to_linear(snr_db)
    
    # Use default phase noise variance if not provided
    if sigma_phi_sq is None:
        # Short-frame approximation from paper
        sigma_phi_sq = 4 * np.pi * utils.Delta_nu * utils.T_frame / 3
    
    # Use default Gamma_eff if not provided
    if Gamma_eff is None:
        B, sigma_eta_sq = utils.compute_bussgang_gain(utils.kappa)
        Gamma_eff = utils.compute_effective_gamma(B, sigma_eta_sq)
    
    # Capacity upper bound
    C = 0.5 * np.log2(1 + rho * np.exp(-sigma_phi_sq) / (1 + Gamma_eff))
    
    return C

def compute_capacity_ceiling(sigma_phi_sq=None, Gamma_eff=None):
    """
    Compute hardware-imposed capacity ceiling from Eq. (67)
    C_sat = lim_{P→∞} C = (1/2) log₂(1 + e^(-σ²_φ)/Γ_eff)
    
    Args:
        sigma_phi_sq: Phase noise variance
        Gamma_eff: Effective distortion factor
        
    Returns:
        C_sat: Capacity ceiling [bits/symbol]
    """
    # Use defaults if not provided
    if sigma_phi_sq is None:
        sigma_phi_sq = 4 * np.pi * utils.Delta_nu * utils.T_frame / 3
    
    if Gamma_eff is None:
        B, sigma_eta_sq = utils.compute_bussgang_gain(utils.kappa)
        Gamma_eff = utils.compute_effective_gamma(B, sigma_eta_sq)
    
    # Capacity ceiling
    C_sat = 0.5 * np.log2(1 + np.exp(-sigma_phi_sq) / Gamma_eff)
    
    return C_sat

def compute_awgn_capacity(snr_db):
    """
    Compute classical AWGN capacity for comparison
    C_AWGN = log₂(1 + SNR)
    
    Args:
        snr_db: SNR in dB
        
    Returns:
        C_awgn: AWGN capacity [bits/symbol]
    """
    snr_linear = utils.db_to_linear(snr_db)
    return np.log2(1 + snr_linear)

def compute_sinr_effective(snr_db, sigma_phi_sq=None, Gamma_eff=None, G_path=None):
    """
    Compute effective SINR with hardware impairments
    
    Args:
        snr_db: Nominal SNR in dB
        sigma_phi_sq: Phase noise variance
        Gamma_eff: Effective distortion factor
        G_path: Path gain (default: computed from nominal parameters)
        
    Returns:
        sinr_eff: Effective SINR (linear)
    """
    # Nominal SNR
    rho = utils.db_to_linear(snr_db)
    
    # Use defaults
    if sigma_phi_sq is None:
        sigma_phi_sq = 4 * np.pi * utils.Delta_nu * utils.T_frame / 3
    
    if Gamma_eff is None:
        B, sigma_eta_sq = utils.compute_bussgang_gain(utils.kappa)
        Gamma_eff = utils.compute_effective_gamma(B, sigma_eta_sq)
    
    if G_path is None:
        G_path = utils.compute_path_loss(utils.R_0, utils.f_c) * \
                 utils.db_to_linear(100)  # Include antenna gains
    
    # Effective SINR accounting for multiplicative phase noise and PA distortion
    sinr_eff = (rho * np.exp(-sigma_phi_sq)) / (1 + rho * Gamma_eff)
    
    return sinr_eff

def sensitivity_analysis_phase_noise(snr_db_range, sigma_phi_sq_range):
    """
    Analyze capacity sensitivity to phase noise variance
    
    Args:
        snr_db_range: Array of SNR values in dB
        sigma_phi_sq_range: Array of phase noise variances
        
    Returns:
        capacity_matrix: 2D array [len(sigma_phi_sq) x len(snr_db)]
    """
    capacity_matrix = np.zeros((len(sigma_phi_sq_range), len(snr_db_range)))
    
    # Get nominal Gamma_eff
    B, sigma_eta_sq = utils.compute_bussgang_gain(utils.kappa)
    Gamma_eff = utils.compute_effective_gamma(B, sigma_eta_sq)
    
    for i, sigma_phi_sq in enumerate(sigma_phi_sq_range):
        for j, snr_db in enumerate(snr_db_range):
            capacity_matrix[i, j] = compute_capacity_upper_bound(
                snr_db, sigma_phi_sq=sigma_phi_sq, Gamma_eff=Gamma_eff
            )
    
    return capacity_matrix

def sensitivity_analysis_distortion(snr_db_range, Gamma_eff_range):
    """
    Analyze capacity sensitivity to effective distortion factor
    
    Args:
        snr_db_range: Array of SNR values in dB
        Gamma_eff_range: Array of Gamma_eff values
        
    Returns:
        capacity_matrix: 2D array [len(Gamma_eff) x len(snr_db)]
    """
    capacity_matrix = np.zeros((len(Gamma_eff_range), len(snr_db_range)))
    
    # Get nominal phase noise
    sigma_phi_sq = 4 * np.pi * utils.Delta_nu * utils.T_frame / 3
    
    for i, Gamma_eff in enumerate(Gamma_eff_range):
        for j, snr_db in enumerate(snr_db_range):
            capacity_matrix[i, j] = compute_capacity_upper_bound(
                snr_db, sigma_phi_sq=sigma_phi_sq, Gamma_eff=Gamma_eff
            )
    
    return capacity_matrix

def compute_capacity_gap(snr_db):
    """
    Compute capacity gap between ideal AWGN and hardware-limited system
    
    Args:
        snr_db: SNR in dB
        
    Returns:
        gap_bits: Capacity gap in bits/symbol
        gap_percent: Capacity gap as percentage
    """
    C_awgn = compute_awgn_capacity(snr_db)
    C_hw = compute_capacity_upper_bound(snr_db)
    
    gap_bits = C_awgn - C_hw
    gap_percent = 100 * gap_bits / C_awgn if C_awgn > 0 else 0
    
    return gap_bits, gap_percent

def validate_capacity_bounds():
    """Run validation tests for capacity computation"""
    print("\n=== Capacity Bounds Validation ===")
    
    # Test at specific SNR values
    snr_test = [10, 20, 30, 40]
    
    print("SNR [dB] | C_AWGN | C_HW | C_sat | Gap [%]")
    print("-" * 50)
    
    for snr in snr_test:
        C_awgn = compute_awgn_capacity(snr)
        C_hw = compute_capacity_upper_bound(snr)
        C_sat = compute_capacity_ceiling()
        gap_bits, gap_percent = compute_capacity_gap(snr)
        
        print(f"{snr:8.0f} | {C_awgn:6.2f} | {C_hw:4.2f} | {C_sat:5.2f} | {gap_percent:7.1f}")
    
    # Verify saturation behavior
    print(f"\nCapacity ceiling: {C_sat:.2f} bits/symbol")
    
    # Check if high-SNR capacity approaches ceiling
    C_high_snr = compute_capacity_upper_bound(60)  # 60 dB SNR
    if np.abs(C_high_snr - C_sat) < 0.01:
        print("✓ Capacity correctly saturates at high SNR")
    else:
        print(f"✗ Warning: High-SNR capacity {C_high_snr:.2f} != ceiling {C_sat:.2f}")
    
    # Test sensitivity
    print("\nPhase noise sensitivity (at 30 dB SNR):")
    sigma_phi_test = [0.01, 0.05, 0.1]
    for sigma_phi in sigma_phi_test:
        C = compute_capacity_upper_bound(30, sigma_phi_sq=sigma_phi)
        print(f"  σ²_φ = {sigma_phi:.2f}: C = {C:.2f} bits/symbol")
    
    print("==================================\n")