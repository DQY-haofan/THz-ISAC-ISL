#!/usr/bin/env python3
"""
capacity_simulation.py

Simulation of communication capacity for THz LEO-ISL ISAC systems.
Demonstrates the hardware-limited capacity ceiling versus ideal AWGN channel.

Based on "Fundamental Limits of THz Inter-Satellite ISAC Under Hardware Impairments"

Author: THz ISL ISAC Simulation Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

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
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 7),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
})

# Use a professional color palette
colors = sns.color_palette("colorblind", 4)

def awgn_capacity(snr_linear: np.ndarray) -> np.ndarray:
    """
    Calculate classical Shannon capacity for AWGN channel.
    
    Implements Eq. (47): C_AWGN = log₂(1 + ρ)
    
    Args:
        snr_linear: Linear SNR values (ρ = P|g|²|B|²/N₀)
        
    Returns:
        Capacity in bits/symbol
    """
    return np.log2(1 + snr_linear)

def hardware_limited_capacity(
    snr_linear: np.ndarray,
    gamma_eff: float,
    sigma_phi_sq: float = None,
    hardware_profile: str = None
) -> np.ndarray:
    """
    Calculate hardware-limited channel capacity.
    
    Implements Eq. (46): C ≤ (1/2) log₂(1 + (ρ·e^(-σ_φ²))/(1 + ρ·Γ_eff))
    
    As ρ → ∞, this saturates at: C_sat = (1/2) log₂(1 + e^(-σ_φ²)/Γ_eff)
    
    Args:
        snr_linear: Linear SNR values (ρ = P|g|²|B|²/N₀)
        gamma_eff: Hardware quality factor
        sigma_phi_sq: Phase noise variance [rad²], if None uses profile default
        hardware_profile: Name of hardware profile (for getting phase noise)
        
    Returns:
        Capacity in bits/symbol
    """
    # Get phase noise variance if not provided
    if sigma_phi_sq is None:
        if hardware_profile is not None:
            profile = HARDWARE_PROFILES[hardware_profile]
            sigma_phi_sq = profile.phase_noise_variance
        else:
            # Default assumption if not specified
            sigma_phi_sq = 0.042  # ~100 kHz linewidth, 100 μs frame
    
    # Phase noise degradation factor
    phase_factor = np.exp(-sigma_phi_sq)
    
    # Hardware-limited SINR
    sinr_eff = (snr_linear * phase_factor) / (1 + snr_linear * gamma_eff)
    
    # Factor of 1/2 accounts for complex-valued channel
    capacity = 0.5 * np.log2(1 + sinr_eff)
    
    return capacity

def calculate_capacity_ceiling(gamma_eff: float, sigma_phi_sq: float) -> float:
    """
    Calculate the asymptotic capacity ceiling.
    
    C_sat = (1/2) log₂(1 + e^(-σ_φ²)/Γ_eff)
    
    Args:
        gamma_eff: Hardware quality factor
        sigma_phi_sq: Phase noise variance [rad²]
        
    Returns:
        Capacity ceiling in bits/symbol
    """
    phase_factor = np.exp(-sigma_phi_sq)
    return 0.5 * np.log2(1 + phase_factor / gamma_eff)

def simulate_capacity_vs_snr():
    """
    Generate Figure 3: Capacity vs. Nominal SNR showing hardware-limited saturation.
    """
    print("=== Generating Capacity vs. SNR Plot ===")
    
    # SNR range (extended to show saturation clearly)
    snr_dB = np.linspace(-10, 50, 200)
    snr_linear = 10 ** (snr_dB / 10)
    
    # Hardware profiles with adjusted Gamma_eff values
    profiles = {
        "High_Performance": {
            "gamma_eff": 0.01,
            "label": "High Performance\n(Γ_eff = 0.01)",
            "color": colors[1],
            "linestyle": '-'
        },
        "SWaP_Efficient": {
            "gamma_eff": 0.045,  # As specified in the prompt
            "label": "SWaP Efficient\n(Γ_eff = 0.045)",
            "color": colors[2], 
            "linestyle": '--'
        }
    }
    
    # Calculate capacities
    capacity_awgn = awgn_capacity(snr_linear)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot AWGN reference
    ax.plot(snr_dB, capacity_awgn, 
            color=colors[0], linewidth=3, linestyle=':',
            label='Ideal AWGN', alpha=0.8)
    
    # Plot hardware-limited cases
    for profile_name, params in profiles.items():
        # Get phase noise from actual profile if available
        if profile_name in HARDWARE_PROFILES:
            sigma_phi_sq = HARDWARE_PROFILES[profile_name].phase_noise_variance
        else:
            sigma_phi_sq = 0.042  # Default
        
        capacity_hw = hardware_limited_capacity(
            snr_linear, 
            params["gamma_eff"],
            sigma_phi_sq
        )
        
        # Calculate ceiling for annotation
        ceiling = calculate_capacity_ceiling(params["gamma_eff"], sigma_phi_sq)
        
        ax.plot(snr_dB, capacity_hw,
                color=params["color"], linewidth=2.5,
                linestyle=params["linestyle"],
                label=params["label"])
        
        # Add ceiling line
        ax.axhline(y=ceiling, color=params["color"], 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Add ceiling annotation
        ax.annotate(f'Ceiling: {ceiling:.2f} bits/symbol',
                    xy=(45, ceiling), xytext=(35, ceiling + 0.5),
                    arrowprops=dict(arrowstyle='->', color=params["color"], alpha=0.7),
                    fontsize=10, color=params["color"])
    
    # Formatting
    ax.set_xlabel('Nominal SNR ρ [dB]', fontsize=14)
    ax.set_ylabel('Capacity [bits/symbol]', fontsize=14)
    ax.set_title('THz ISL Channel Capacity: Hardware Limitations vs. Ideal AWGN',
                 fontsize=16, pad=15)
    
    # Grid and limits
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(-10, 50)
    ax.set_ylim(0, 8)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, 
              shadow=True, ncol=1, fontsize=11)
    
    # Add text box explaining the phenomenon
    textstr = (
        'Key Insight:\n'
        '• AWGN capacity grows unbounded with power\n'
        '• Hardware impairments create capacity ceilings\n'
        '• Higher Γ_eff → Lower ceiling'
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add shaded regions to highlight saturation
    ax.axvspan(30, 50, alpha=0.1, color='gray', label='_nolegend_')
    ax.text(40, 0.5, 'Hardware-limited\nregime', ha='center', fontsize=10,
            style='italic', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('capacity_vs_snr.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('capacity_vs_snr.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical results
    print("\n=== Capacity Ceiling Analysis ===")
    for profile_name, params in profiles.items():
        if profile_name in HARDWARE_PROFILES:
            sigma_phi_sq = HARDWARE_PROFILES[profile_name].phase_noise_variance
        else:
            sigma_phi_sq = 0.042
            
        ceiling = calculate_capacity_ceiling(params["gamma_eff"], sigma_phi_sq)
        
        # Calculate SNR where capacity reaches 95% of ceiling
        cap_at_30dB = hardware_limited_capacity(
            np.array([1000]), params["gamma_eff"], sigma_phi_sq
        )[0]
        saturation_pct = (cap_at_30dB / ceiling) * 100
        
        print(f"\n{params['label'].replace(chr(10), ' ')}:")
        print(f"  Gamma_eff: {params['gamma_eff']:.3f}")
        print(f"  Phase noise variance: {sigma_phi_sq:.4f} rad²")
        print(f"  Capacity ceiling: {ceiling:.3f} bits/symbol")
        print(f"  At 30 dB SNR: {cap_at_30dB:.3f} bits/symbol ({saturation_pct:.1f}% of ceiling)")
        print(f"  Equivalent to: {2**ceiling:.1f}-QAM maximum constellation")

def plot_capacity_components():
    """
    Additional plot showing the breakdown of capacity limitations.
    """
    print("\n=== Generating Capacity Components Plot ===")
    
    # Fixed parameters
    gamma_eff = 0.045  # SWaP Efficient
    sigma_phi_sq = 0.042
    
    # SNR range
    snr_dB = np.linspace(-10, 40, 100)
    snr_linear = 10 ** (snr_dB / 10)
    
    # Calculate different scenarios
    # 1. Ideal AWGN
    cap_awgn = awgn_capacity(snr_linear)
    
    # 2. With PA distortion only (no phase noise)
    cap_pa_only = hardware_limited_capacity(snr_linear, gamma_eff, 0)
    
    # 3. With phase noise only (no PA distortion)
    phase_factor = np.exp(-sigma_phi_sq)
    cap_pn_only = 0.5 * np.log2(1 + snr_linear * phase_factor)
    
    # 4. With both impairments
    cap_both = hardware_limited_capacity(snr_linear, gamma_eff, sigma_phi_sq)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot all curves
    ax.plot(snr_dB, cap_awgn, 'k:', linewidth=3, label='Ideal AWGN', alpha=0.8)
    ax.plot(snr_dB, cap_pn_only, 'b--', linewidth=2, label='Phase noise only', alpha=0.7)
    ax.plot(snr_dB, cap_pa_only, 'r--', linewidth=2, label='PA distortion only', alpha=0.7)
    ax.plot(snr_dB, cap_both, 'g-', linewidth=3, label='Both impairments')
    
    # Fill areas to show loss
    ax.fill_between(snr_dB, cap_pn_only, cap_awgn, alpha=0.2, color='blue', 
                    label='Loss from phase noise')
    ax.fill_between(snr_dB, cap_both, cap_pn_only, alpha=0.2, color='red',
                    label='Additional loss from PA')
    
    # Formatting
    ax.set_xlabel('Nominal SNR [dB]', fontsize=14)
    ax.set_ylabel('Capacity [bits/symbol]', fontsize=14)
    ax.set_title('Impact of Individual Hardware Impairments on Capacity\n(SWaP Efficient Profile)',
                 fontsize=16, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-10, 40)
    ax.set_ylim(0, 6)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('capacity_components.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('capacity_components.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run capacity simulations."""
    print("=== THz ISL ISAC Capacity Simulation ===")
    print(f"Analyzing hardware-limited capacity ceiling effect")
    
    # Generate main capacity plot
    simulate_capacity_vs_snr()
    
    # Generate component breakdown plot
    plot_capacity_components()
    
    print("\n=== Simulation Complete ===")
    print("Output files:")
    print("  - capacity_vs_snr.pdf/png")
    print("  - capacity_components.pdf/png")

if __name__ == "__main__":
    main()