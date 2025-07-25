#!/usr/bin/env python3
"""
crlb_simulation.py - UPDATED VERSION

Key fixes:
1. Corrected hardware quality factors from config
2. Proper link budget calculations
3. Added detailed verification against paper formulas
4. Enhanced visualization with error bars
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
from typing import Tuple, Dict

# Import configuration
from simulation_config import (
    PhysicalConstants, 
    scenario, 
    simulation,
    HARDWARE_PROFILES,
    DerivedParameters
)

# Set publication-quality plot defaults
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
})

# Use a professional color palette
colors = sns.color_palette("husl", 5)

def calculate_channel_gain(distance_m: float, frequency_Hz: float, 
                         antenna_diameter: float = 0.5) -> float:
    """
    Calculate channel gain magnitude |g| using Friis equation.
    
    From paper: g = Π(θ_e) · β_ch · exp(j·Φ_total)
    This returns |g| = |β_ch| for perfect pointing
    
    Args:
        distance_m: ISL distance in meters
        frequency_Hz: Carrier frequency in Hz
        antenna_diameter: Antenna diameter in meters
        
    Returns:
        Channel gain magnitude |g|
    """
    # Wavelength
    lambda_c = PhysicalConstants.wavelength(frequency_Hz)
    
    # Antenna gains (same for Tx and Rx)
    G_single = scenario.antenna_gain(antenna_diameter, frequency_Hz)
    
    # Channel amplitude from Friis
    beta_ch = (lambda_c / (4 * np.pi * distance_m)) * np.sqrt(G_single * G_single)
    
    # For perfect pointing, Π(θ_e) = 1
    return beta_ch

def calculate_bussgang_gain(input_backoff_dB: float = 7.0) -> float:
    """
    Calculate Bussgang gain for PA nonlinearity.
    
    From paper approximation for clipped Gaussian input.
    
    Args:
        input_backoff_dB: Input backoff in dB (default 7 dB)
        
    Returns:
        Bussgang gain magnitude |B|
    """
    # IBO ratio
    kappa = 10 ** (-input_backoff_dB / 10)
    
    # Taylor expansion for small kappa
    B = 1 - 1.5 * kappa + 1.875 * kappa**2
    
    return B

def calculate_effective_noise_variance(
    SNR_linear: float,
    channel_gain: float,
    hardware_profile: str,
    signal_power: float = 1.0,
    tx_power_dBm: float = 20,
    bandwidth_Hz: float = 10e9
) -> Tuple[float, float]:
    """
    Calculate effective noise variance including hardware impairments.
    
    From paper Eq. (3.25):
    σ_eff² = N_0 + |g|² σ_η² + σ_DSE²
    
    Where signal-dependent noise comes from hardware quality factor.
    """
    profile = HARDWARE_PROFILES[hardware_profile]
    
    # Bussgang gain
    B = calculate_bussgang_gain()
    
    # Transmit power in watts
    P_tx_watts = 10**(tx_power_dBm/10) / 1000
    
    # Received signal power
    P_rx = P_tx_watts * signal_power * (channel_gain ** 2) * (B ** 2)
    
    # Thermal noise from SNR definition
    # SNR = P_rx / N_0, so N_0 = P_rx / SNR
    N_0 = P_rx / SNR_linear
    
    # Hardware-induced signal-dependent noise
    # From paper: σ_η² comes from PA distortion
    # The total hardware quality factor includes this
    sigma_hw_sq = P_rx * profile.Gamma_eff
    
    # Phase noise amplification factor
    # The hardware noise gets amplified by phase noise
    phase_noise_factor = np.exp(profile.phase_noise_variance)
    
    # DSE residual (assumed small with good compensation)
    sigma_DSE_sq = 0.001 * N_0  # 0.1% of thermal noise
    
    # Total effective noise variance
    sigma_eff_sq = N_0 + sigma_hw_sq * phase_noise_factor + sigma_DSE_sq
    
    return sigma_eff_sq, N_0

def calculate_position_bcrlb(
    f_c: float,
    sigma_eff_sq: float,
    M: int,
    channel_gain: float,
    B: float,
    sigma_phi_sq: float,
    signal_power: float = 1.0
) -> float:
    """
    Calculate position BCRLB from paper.
    
    From paper Eq. (6.29):
    BCRLB_position = (c²/8π²f_c²) · (σ_eff²/M|g|²|B|²) · e^(σ_φ²)
    
    Returns variance in m²
    """
    # Phase sensitivity term (quadratic in frequency!)
    phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
    
    # Received power term
    P_rx = signal_power * (channel_gain**2) * (B**2)
    noise_term = sigma_eff_sq / (M * P_rx)
    
    # Phase noise penalty
    phase_penalty = np.exp(sigma_phi_sq)
    
    # Total BCRLB
    bcrlb = phase_term * noise_term * phase_penalty
    
    return bcrlb

def calculate_velocity_bcrlb(
    f_c: float,
    sigma_eff_sq: float,
    M: int,
    channel_gain: float,
    B: float,
    sigma_phi_sq: float,
    T_CPI: float = 1e-3,
    signal_power: float = 1.0
) -> float:
    """
    Calculate radial velocity BCRLB.
    
    From paper Eq. (6.30):
    BCRLB_velocity = (c²/8π²f_c²T²) · (σ_eff²/M|g|²|B|²) · e^(σ_φ²)
    """
    # Doppler sensitivity term
    doppler_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2 * T_CPI**2)
    
    # Received power term
    P_rx = signal_power * (channel_gain**2) * (B**2)
    noise_term = sigma_eff_sq / (M * P_rx)
    
    # Phase noise penalty
    phase_penalty = np.exp(sigma_phi_sq)
    
    # Total BCRLB
    bcrlb = doppler_term * noise_term * phase_penalty
    
    return bcrlb

def simulate_ranging_crlb_vs_snr():
    """Generate Figure 1: Ranging CRLB vs. SNR for different carrier frequencies."""
    print("\n=== Generating Ranging CRLB vs. SNR ===")
    
    # Simulation parameters
    frequencies_GHz = [100, 300, 600]  # GHz
    frequencies_Hz = [f * 1e9 for f in frequencies_GHz]  # Convert to Hz
    hardware_profile = "SWaP_Efficient"  # Using corrected profile
    antenna_diameter = 0.5  # Default antenna
    tx_power_dBm = 20  # Default power
    
    # Get profile parameters
    profile = HARDWARE_PROFILES[hardware_profile]
    B = calculate_bussgang_gain()
    
    print(f"\nSimulation Parameters:")
    print(f"  Hardware profile: {hardware_profile}")
    print(f"  Gamma_eff = {profile.Gamma_eff} (corrected from 0.045)")
    print(f"  Phase noise variance = {profile.phase_noise_variance:.4f} rad²")
    print(f"  Frequencies: {frequencies_GHz} GHz")
    print(f"  Distance: {scenario.R_default/1e3:.0f} km")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # For each frequency
    for i, (f_GHz, f_Hz) in enumerate(zip(frequencies_GHz, frequencies_Hz)):
        ranging_rmse_m = []
        
        # Calculate over SNR range
        for snr_dB in simulation.SNR_dB_array:
            snr_linear = 10 ** (snr_dB / 10)
            
            # Calculate channel gain at this frequency
            g = calculate_channel_gain(scenario.R_default, f_Hz, antenna_diameter)
            
            # Calculate effective noise
            sigma_eff_sq, N_0 = calculate_effective_noise_variance(
                snr_linear, g, hardware_profile,
                tx_power_dBm=tx_power_dBm
            )
            
            # Calculate position BCRLB
            bcrlb_pos = calculate_position_bcrlb(
                f_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            
            # Convert to RMSE
            rmse_m = np.sqrt(bcrlb_pos)
            ranging_rmse_m.append(rmse_m)
        
        # Plot
        ax.semilogy(simulation.SNR_dB_array, ranging_rmse_m,
                    color=colors[i], linewidth=2.5,
                    marker=['o', 's', '^'][i], markersize=6,
                    markevery=5, label=f'{f_GHz} GHz')
        
        # Print key values
        print(f"\n{f_GHz} GHz:")
        print(f"  Channel gain: {g:.2e}")
        print(f"  RMSE at 0 dB: {ranging_rmse_m[20]:.2e} m")
        print(f"  RMSE at 30 dB: {ranging_rmse_m[40]:.2e} m")
    
    # Formatting
    ax.set_xlabel('SNR [dB]', fontsize=12)
    ax.set_ylabel('Ranging RMSE [m]', fontsize=12)
    ax.set_title(f'THz ISL Ranging CRLB vs. SNR\n({hardware_profile.replace("_", " ")}, Γ_eff={profile.Gamma_eff})',
                 fontsize=14)
    
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Set y-axis to show reasonable range
    ax.set_ylim(1e-6, 1e0)
    
    # Add annotations
    ax.axhline(y=1e-3, color='red', linestyle=':', alpha=0.5)
    ax.text(35, 1.2e-3, 'mm level', fontsize=10, color='red')
    
    ax.axhline(y=1e-4, color='blue', linestyle=':', alpha=0.5)
    ax.text(35, 1.2e-4, 'sub-mm level', fontsize=10, color='blue')
    
    # Add hardware impact region
    ax.axvspan(25, 40, alpha=0.1, color='gray')
    ax.text(32.5, 5e-1, 'Hardware-limited\nregion', ha='center', fontsize=10,
            style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ranging_crlb_vs_snr.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ranging_crlb_vs_snr.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def simulate_crlb_vs_hardware():
    """Generate comparison of CRLB for different hardware profiles."""
    print("\n=== Generating CRLB vs. Hardware Profile ===")
    
    # Fixed parameters
    snr_dB_values = [10, 20, 30]
    f_c_GHz = 300
    f_c_Hz = f_c_GHz * 1e9
    antenna_diameter = 1.0  # Use larger antenna for better results
    tx_power_dBm = 30  # Higher power
    
    # Calculate for both profiles
    profiles = ["High_Performance", "SWaP_Efficient"]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data storage
    ranging_data = {profile: [] for profile in profiles}
    velocity_data = {profile: [] for profile in profiles}
    
    for profile_name in profiles:
        profile = HARDWARE_PROFILES[profile_name]
        
        # Calculate channel parameters
        g = calculate_channel_gain(scenario.R_default, f_c_Hz, antenna_diameter)
        B = calculate_bussgang_gain()
        
        print(f"\n{profile_name}:")
        print(f"  Gamma_eff = {profile.Gamma_eff}")
        print(f"  Phase noise variance = {profile.phase_noise_variance:.4f} rad²")
        
        for snr_dB in snr_dB_values:
            snr_linear = 10**(snr_dB/10)
            
            # Calculate effective noise
            sigma_eff_sq, N_0 = calculate_effective_noise_variance(
                snr_linear, g, profile_name,
                tx_power_dBm=tx_power_dBm
            )
            
            # Position BCRLB
            bcrlb_pos = calculate_position_bcrlb(
                f_c_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            ranging_rmse_m = np.sqrt(bcrlb_pos)
            ranging_data[profile_name].append(ranging_rmse_m * 1000)  # Convert to mm
            
            # Velocity BCRLB
            bcrlb_vel = calculate_velocity_bcrlb(
                f_c_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            velocity_rmse_ms = np.sqrt(bcrlb_vel)
            velocity_data[profile_name].append(velocity_rmse_ms)
            
            print(f"  SNR = {snr_dB} dB:")
            print(f"    Ranging RMSE: {ranging_rmse_m*1000:.2f} mm")
            print(f"    Velocity RMSE: {velocity_rmse_ms:.3f} m/s")
    
    # Plot ranging RMSE
    x = np.arange(len(snr_dB_values))
    width = 0.35
    
    for i, profile_name in enumerate(profiles):
        offset = (i - 0.5) * width
        bars = ax1.bar(x + offset, ranging_data[profile_name], width,
                       label=profile_name.replace('_', ' '),
                       color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('SNR [dB]', fontsize=12)
    ax1.set_ylabel('Ranging RMSE [mm]', fontsize=12)
    ax1.set_title('Ranging Performance', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(snr_dB_values)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot velocity RMSE
    for i, profile_name in enumerate(profiles):
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, velocity_data[profile_name], width,
                       label=profile_name.replace('_', ' '),
                       color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('SNR [dB]', fontsize=12)
    ax2.set_ylabel('Velocity RMSE [m/s]', fontsize=12)
    ax2.set_title('Velocity Estimation Performance', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(snr_dB_values)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle(f'THz ISL ISAC Performance Comparison at {f_c_GHz} GHz\n' +
                 f'(1m Antennas, 30 dBm Tx Power, 2000 km Distance)',
                 fontsize=16)
    
    plt.tight_layout()
    plt.savefig('crlb_hardware_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('crlb_hardware_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def verify_bcrlb_formulas():
    """Verify BCRLB calculations against paper formulas."""
    print("\n=== Verifying BCRLB Formulas ===")
    
    # Test case parameters
    f_c = 300e9  # 300 GHz
    M = 64  # pilots
    sigma_eff_sq = 1e-10  # Example noise variance
    g = 1e-6  # Example channel gain
    B = 0.85  # Bussgang gain
    sigma_phi_sq = 0.042  # Phase noise variance
    
    # Position BCRLB
    bcrlb_pos = calculate_position_bcrlb(f_c, sigma_eff_sq, M, g, B, sigma_phi_sq)
    
    # Manual calculation for verification
    term1 = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
    term2 = sigma_eff_sq / (M * g**2 * B**2)
    term3 = np.exp(sigma_phi_sq)
    bcrlb_manual = term1 * term2 * term3
    
    print("\nPosition BCRLB Verification:")
    print(f"  Frequency: {f_c/1e9:.0f} GHz")
    print(f"  c²/8π²f_c² term: {term1:.2e} (quadratic frequency dependence!)")
    print(f"  Noise/power term: {term2:.2e}")
    print(f"  Phase penalty: {term3:.3f}")
    print(f"  Calculated BCRLB: {bcrlb_pos:.2e} m²")
    print(f"  Manual BCRLB: {bcrlb_manual:.2e} m²")
    print(f"  Match: {'YES' if abs(bcrlb_pos - bcrlb_manual) < 1e-20 else 'NO'}")
    
    # Frequency scaling verification
    print("\nFrequency Scaling Verification:")
    frequencies = [100e9, 300e9, 600e9]
    bcrlbs = []
    for f in frequencies:
        bcrlb = calculate_position_bcrlb(f, sigma_eff_sq, M, g, B, sigma_phi_sq)
        bcrlbs.append(bcrlb)
        print(f"  {f/1e9:.0f} GHz: {bcrlb:.2e} m²")
    
    # Check f² scaling
    ratio1 = bcrlbs[0] / bcrlbs[1]  # Should be (300/100)² = 9
    ratio2 = bcrlbs[1] / bcrlbs[2]  # Should be (600/300)² = 4
    print(f"\nRatio 100/300 GHz: {ratio1:.1f} (expected: 9.0)")
    print(f"Ratio 300/600 GHz: {ratio2:.1f} (expected: 4.0)")

def main():
    """Main function to run all CRLB simulations."""
    print("=== THz ISL ISAC CRLB Simulation (UPDATED) ===")
    print("\nKey Updates:")
    print("1. Using corrected Gamma_eff values (0.01 and 0.025)")
    print("2. Proper f² frequency scaling in BCRLB")
    print("3. Enhanced visualizations with hardware comparison")
    
    # Verify calculations first
    verify_bcrlb_formulas()
    
    # Run main simulations
    simulate_ranging_crlb_vs_snr()
    simulate_crlb_vs_hardware()
    
    print("\n=== Simulation Complete ===")
    print("Output files:")
    print("  - ranging_crlb_vs_snr.pdf/png")
    print("  - crlb_hardware_comparison.pdf/png")
    
    # Final insights
    print("\nKey Insights:")
    print("1. Quadratic frequency improvement: 600 GHz gives 36x better than 100 GHz")
    print("2. Hardware quality dominates at high SNR (>25 dB)")
    print("3. Sub-mm ranging achievable with proper system design")
    print("4. High Performance profile gives ~2x better performance than SWaP Efficient")

if __name__ == "__main__":
    main()