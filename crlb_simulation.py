#!/usr/bin/env python3
"""
crlb_simulation.py - FINAL FIXED VERSION

Fixed issues:
1. Ensured all calculations use SI units (m, m/s, Hz)
2. Using correct phase noise variance from config
3. Proper unit conversions where needed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
                         antenna_gain_linear: float = None) -> float:
    """
    Calculate channel gain using Friis equation.
    
    Args:
        distance_m: ISL distance in meters (SI unit)
        frequency_Hz: Carrier frequency in Hz (SI unit)
        antenna_gain_linear: Combined antenna gain (G_tx * G_rx), if None uses scenario default
        
    Returns:
        Channel gain magnitude |g|
    """
    if antenna_gain_linear is None:
        # Use default antenna gain (same for Tx and Rx)
        antenna_gain_linear = scenario.antenna_gain ** 2
    
    lambda_c = PhysicalConstants.wavelength(frequency_Hz)
    beta_ch = (lambda_c / (4 * np.pi * distance_m)) * np.sqrt(antenna_gain_linear)
    
    return beta_ch

def calculate_bussgang_gain(input_backoff_dB: float = 7.0) -> float:
    """
    Calculate Bussgang gain for PA nonlinearity.
    
    Args:
        input_backoff_dB: Input backoff in dB (default 7 dB)
        
    Returns:
        Bussgang gain magnitude |B|
    """
    # From manuscript approximation for typical operating point
    kappa = 10 ** (-input_backoff_dB / 10)  # IBO ratio
    
    # Taylor expansion approximation for small kappa
    B = 1 - 1.5 * kappa + 1.875 * kappa**2
    
    return B

def calculate_effective_noise_variance(
    SNR_linear: float,
    channel_gain: float,
    hardware_profile: str,
    signal_power: float = 1.0
) -> Tuple[float, float]:
    """
    Calculate effective noise variance including hardware impairments.
    
    修正：确保硬件噪声正确计入
    """
    profile = HARDWARE_PROFILES[hardware_profile]
    
    # Calculate thermal noise from SNR definition
    B = calculate_bussgang_gain()
    N_0 = signal_power * (channel_gain ** 2) * (B ** 2) / SNR_linear
    
    # 关键修正：使用完整的Gamma_eff而不仅仅是Gamma_PA
    # 硬件引起的信号相关噪声功率
    P_rx = signal_power * (channel_gain ** 2) * (B ** 2)  # 接收信号功率
    sigma_hw_sq = P_rx * profile.Gamma_eff  # 使用总的Gamma_eff
    
    # Phase noise effect (multiplicative factor)
    phase_noise_factor = np.exp(profile.phase_noise_variance)
    
    # DSE residual (from manuscript, negligible after compensation)
    sigma_DSE_sq = 0.001 / SNR_linear
    
    # Total effective noise variance (Eq. (24) in manuscript)
    # 修正：硬件噪声应该乘以相位噪声因子
    sigma_eff_sq = N_0 + sigma_hw_sq * phase_noise_factor + sigma_DSE_sq
    
    return sigma_eff_sq, N_0

def calculate_position_bcrlb(
    f_c: float,
    sigma_eff_sq: float,
    M: int,
    channel_gain: float,
    B: float,
    sigma_phi_sq: float
) -> float:
    """
    Calculate Position Bayesian Cramér-Rao Lower Bound.
    
    Implements Eq. (49) from the manuscript:
    BCRLB_position = (c²/(8π²f_c²)) * (σ_eff²/(M|g|²|B|²)) * e^(σ_φ²)
    
    Args:
        f_c: Carrier frequency [Hz] (SI unit)
        sigma_eff_sq: Effective noise variance
        M: Number of pilot symbols
        channel_gain: Channel gain magnitude |g|
        B: Bussgang gain magnitude |B|
        sigma_phi_sq: Phase noise variance [rad²]
        
    Returns:
        Position BCRLB [m²] (SI unit)
    """
    # First term: fundamental ranging resolution
    term1 = PhysicalConstants.c ** 2 / (8 * np.pi**2 * f_c**2)
    
    # Second term: SNR-dependent factor
    term2 = sigma_eff_sq / (M * channel_gain**2 * B**2)
    
    # Third term: phase noise penalty
    term3 = np.exp(sigma_phi_sq)
    
    bcrlb_position = term1 * term2 * term3
    
    return bcrlb_position

def simulate_ranging_crlb_vs_snr():
    """
    Generate Figure 1: Ranging CRLB vs. SNR for different carrier frequencies.
    Shows the f_c² scaling and hardware-imposed performance floor.
    """
    print("Generating Figure 1: Ranging CRLB vs. SNR...")
    
    # Simulation parameters
    frequencies_GHz = [100, 300, 600]  # GHz
    frequencies_Hz = [f * 1e9 for f in frequencies_GHz]  # Convert to Hz (SI unit)
    hardware_profile = "SWaP_Efficient"  # Use standard profile for this plot
    
    # Get profile parameters
    profile = HARDWARE_PROFILES[hardware_profile]
    B = calculate_bussgang_gain()
    
    # Initialize results storage
    results = {f: [] for f in frequencies_GHz}
    
    # Iterate over SNR range
    for snr_dB in simulation.SNR_dB_array:
        snr_linear = 10 ** (snr_dB / 10)
        
        for f_GHz, f_Hz in zip(frequencies_GHz, frequencies_Hz):
            # Calculate channel gain at this frequency
            # Using SI units: distance in meters
            g = calculate_channel_gain(scenario.R_default, f_Hz)
            
            # Calculate effective noise
            sigma_eff_sq, _ = calculate_effective_noise_variance(
                snr_linear, g, hardware_profile
            )
            
            # Calculate position BCRLB
            bcrlb_pos = calculate_position_bcrlb(
                f_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            
            # Convert to ranging RMSE (square root for standard deviation)
            ranging_rmse_m = np.sqrt(bcrlb_pos)
            
            results[f_GHz].append(ranging_rmse_m)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (f_GHz, ranging_rmse) in enumerate(results.items()):
        ax.semilogy(simulation.SNR_dB_array, ranging_rmse, 
                   color=colors[i], linewidth=2.5,
                   label=f'{f_GHz} GHz', marker='o', markevery=5,
                   markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # Add theoretical f_c² scaling reference lines
    ref_snr = 20  # Reference SNR for scaling demonstration
    ref_idx = np.argmin(np.abs(simulation.SNR_dB_array - ref_snr))
    for i, f_GHz in enumerate(frequencies_GHz):
        if i > 0:
            # Show f_c² improvement from 100 GHz baseline
            scaling = (frequencies_GHz[0] / f_GHz) ** 2
            ref_value = results[frequencies_GHz[0]][ref_idx] * scaling
            ax.plot([ref_snr-5, ref_snr+5], [ref_value, ref_value],
                   '--', color=colors[i], alpha=0.5, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('SNR [dB]', fontsize=12)
    ax.set_ylabel('Ranging RMSE [m]', fontsize=12)
    ax.set_title('THz ISL Ranging Performance vs. SNR\n' + 
                f'(Hardware: {hardware_profile}, Distance: {scenario.R_default/1e3:.0f} km)',
                fontsize=14, pad=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits to show performance floor
    ax.set_xlim(simulation.SNR_dB_min, simulation.SNR_dB_max)
    ax.set_ylim(1e-4, 1e2)
    
    # Add annotation for hardware floor
    ax.annotate('Hardware-limited\nperformance floor', 
                xy=(35, results[600][-1]), xytext=(25, 5e-3),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('ranging_crlb_vs_snr.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ranging_crlb_vs_snr.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure 1 saved. Performance floor at high SNR: ~{results[600][-1]:.2e} m")

def simulate_ranging_crlb_vs_hardware():
    """
    Generate Figure 2: Ranging CRLB vs. Hardware Profile at fixed high SNR.
    Shows the impact of hardware quality on best-case sensing accuracy.
    """
    print("\nGenerating Figure 2: Ranging CRLB vs. Hardware Profile...")
    
    # Fixed parameters
    snr_dB = 30  # High SNR to see hardware limitations
    snr_linear = 10 ** (snr_dB / 10)
    f_c_GHz = 300
    f_c_Hz = f_c_GHz * 1e9  # Convert to Hz (SI unit)
    
    # Calculate for both hardware profiles
    profiles = ["High_Performance", "SWaP_Efficient"]
    ranging_rmse_results = []
    component_contributions = []
    
    for profile_name in profiles:
        profile = HARDWARE_PROFILES[profile_name]
        
        # Calculate channel parameters
        # Using SI units: distance in meters
        g = calculate_channel_gain(scenario.R_default, f_c_Hz)
        B = calculate_bussgang_gain()
        
        # Calculate effective noise
        sigma_eff_sq, N_0 = calculate_effective_noise_variance(
            snr_linear, g, profile_name
        )
        
        # Calculate position BCRLB
        bcrlb_pos = calculate_position_bcrlb(
            f_c_Hz, sigma_eff_sq, simulation.n_pilots,
            g, B, profile.phase_noise_variance
        )
        
        ranging_rmse_m = np.sqrt(bcrlb_pos)
        ranging_rmse_results.append(ranging_rmse_m)
        
        # Store component contributions for detailed analysis
        component_contributions.append({
            'Gamma_PA': profile.Gamma_PA,
            'Gamma_LO': profile.Gamma_LO,
            'Gamma_ADC': profile.Gamma_ADC,
            'Gamma_eff': profile.Gamma_eff,
            'Phase_noise_var': profile.phase_noise_variance,
            'Thermal_noise': N_0,
            'Effective_noise': sigma_eff_sq
        })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Ranging RMSE comparison
    x_pos = np.arange(len(profiles))
    bars1 = ax1.bar(x_pos, np.array(ranging_rmse_results) * 1000,  # Convert to mm
                     color=[colors[0], colors[2]], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, ranging_rmse_results)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*1000:.2f} mm', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Ranging RMSE [mm]', fontsize=12)
    ax1.set_title(f'Ranging Performance at {snr_dB} dB SNR\n({f_c_GHz} GHz, {scenario.R_default/1e3:.0f} km)',
                  fontsize=14, pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(profiles, fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, max(ranging_rmse_results) * 1200)
    
    # Add improvement factor annotation
    improvement = ranging_rmse_results[1] / ranging_rmse_results[0]
    ax1.annotate(f'{improvement:.1f}x worse', 
                xy=(0.5, max(ranging_rmse_results) * 600), 
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    # Subplot 2: Hardware quality factor breakdown
    width = 0.35
    x_pos2 = np.arange(len(profiles))
    
    # Stack the component contributions
    gamma_pa = [comp['Gamma_PA'] for comp in component_contributions]
    gamma_lo = [comp['Gamma_LO'] for comp in component_contributions]
    gamma_adc = [comp['Gamma_ADC'] for comp in component_contributions]
    
    # Convert to percentage of total
    gamma_total = [comp['Gamma_eff'] for comp in component_contributions]
    pa_pct = [pa/tot * 100 for pa, tot in zip(gamma_pa, gamma_total)]
    lo_pct = [lo/tot * 100 for lo, tot in zip(gamma_lo, gamma_total)]
    adc_pct = [adc/tot * 100 for adc, tot in zip(gamma_adc, gamma_total)]
    
    # Create stacked bar chart
    p1 = ax2.bar(x_pos2, pa_pct, width, label='PA', color=colors[1], alpha=0.8)
    p2 = ax2.bar(x_pos2, lo_pct, width, bottom=pa_pct, label='LO', color=colors[3], alpha=0.8)
    p3 = ax2.bar(x_pos2, adc_pct, width, bottom=np.array(pa_pct)+np.array(lo_pct), 
                 label='ADC', color=colors[4], alpha=0.8)
    
    # Add total Gamma_eff values as text
    for i, (x, gamma) in enumerate(zip(x_pos2, gamma_total)):
        ax2.text(x, 105, f'Γ_eff = {gamma:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Component Contribution [%]', fontsize=12)
    ax2.set_title('Hardware Quality Factor Breakdown', fontsize=14, pad=10)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(profiles, fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0, 115)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('ranging_crlb_vs_hardware.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ranging_crlb_vs_hardware.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print(f"\nDetailed Results at {snr_dB} dB SNR:")
    for i, (profile_name, rmse, comp) in enumerate(zip(profiles, ranging_rmse_results, component_contributions)):
        print(f"\n{profile_name}:")
        print(f"  Ranging RMSE: {rmse*1000:.3f} mm ({rmse:.3e} m)")
        print(f"  Gamma_eff: {comp['Gamma_eff']:.4f}")
        print(f"  Phase noise variance: {comp['Phase_noise_var']:.4f} rad²")
        print(f"  Component breakdown:")
        print(f"    - PA:  {comp['Gamma_PA']:.2e} ({comp['Gamma_PA']/comp['Gamma_eff']*100:.1f}%)")
        print(f"    - LO:  {comp['Gamma_LO']:.2e} ({comp['Gamma_LO']/comp['Gamma_eff']*100:.1f}%)")
        print(f"    - ADC: {comp['Gamma_ADC']:.2e} ({comp['Gamma_ADC']/comp['Gamma_eff']*100:.2f}%)")

def main():
    """Main function to run all simulations."""
    print("=== THz ISL ISAC CRLB Simulation ===")
    print(f"Configuration: {simulation.n_pilots} pilots, {scenario.R_default/1e3:.0f} km distance")
    print(f"Hardware profiles: {list(HARDWARE_PROFILES.keys())}")
    
    # Verify units
    print("\nUnit Check:")
    print(f"  Distance: {scenario.R_default} m (SI unit)")
    print(f"  Velocity: {scenario.v_rel_default} m/s (SI unit)")
    print(f"  Speed of light: {PhysicalConstants.c} m/s (SI unit)")
    
    # Run simulations
    simulate_ranging_crlb_vs_snr()
    simulate_ranging_crlb_vs_hardware()
    
    print("\n=== Simulation Complete ===")
    print("Output files:")
    print("  - ranging_crlb_vs_snr.pdf/png")
    print("  - ranging_crlb_vs_hardware.pdf/png")

if __name__ == "__main__":
    main()