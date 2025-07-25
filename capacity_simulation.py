#!/usr/bin/env python3
"""
capacity_simulation.py - FINAL FIXED VERSION

Fixed issues:
1. Removed 1/2 factor for complex baseband channel
2. Ensured all units are in SI (m, m/s, Hz)
3. Added hardware component visualization
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
    For complex baseband: C_AWGN = log₂(1 + ρ)
    
    Args:
        snr_linear: Linear SNR values (ρ = P|g|²|B|²/N₀)
        
    Returns:
        Capacity in bits/symbol
    """
    return np.log2(1 + snr_linear)

def hardware_limited_capacity(
    snr_linear: np.ndarray,
    gamma_eff: float,
    sigma_phi_sq: float
) -> np.ndarray:
    """
    Calculate hardware-limited channel capacity.
    
    For complex baseband channel:
    C = log₂(1 + (ρ·e^(-σ_φ²))/(1 + ρ·Γ_eff))
    
    NO 1/2 factor for complex channel!
    
    Args:
        snr_linear: Linear SNR values (ρ = P|g|²|B|²/N₀)
        gamma_eff: Hardware quality factor
        sigma_phi_sq: Phase noise variance [rad²]
        
    Returns:
        Capacity in bits/symbol
    """
    # Phase noise degradation factor
    phase_factor = np.exp(-sigma_phi_sq)
    
    # Hardware-limited SINR
    sinr_eff = (snr_linear * phase_factor) / (1 + snr_linear * gamma_eff)
    
    # Complex baseband capacity (no 1/2 factor)
    capacity = np.log2(1 + sinr_eff)
    
    return capacity

def calculate_capacity_ceiling(gamma_eff: float, sigma_phi_sq: float) -> float:
    """
    Calculate the asymptotic capacity ceiling.
    
    C_sat = log₂(1 + e^(-σ_φ²)/Γ_eff)
    
    Args:
        gamma_eff: Hardware quality factor
        sigma_phi_sq: Phase noise variance [rad²]
        
    Returns:
        Capacity ceiling in bits/symbol
    """
    phase_factor = np.exp(-sigma_phi_sq)
    return np.log2(1 + phase_factor / gamma_eff)

def simulate_capacity_vs_snr():
    """
    Generate Figure 3: Capacity vs. Nominal SNR showing hardware-limited saturation.
    """
    print("=== Generating Capacity vs. SNR Plot ===")
    
    # SNR range (extended to show saturation clearly)
    snr_dB = np.linspace(-10, 50, 200)
    snr_linear = 10 ** (snr_dB / 10)
    
    # Calculate capacities
    capacity_awgn = awgn_capacity(snr_linear)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot AWGN reference
    ax.plot(snr_dB, capacity_awgn, 
            color=colors[0], linewidth=3, linestyle=':',
            label='Ideal AWGN', alpha=0.8)
    
    # Plot hardware-limited cases using configuration values
    profile_colors = [colors[1], colors[2]]
    profile_styles = ['-', '--']
    
    for i, (name, profile) in enumerate(HARDWARE_PROFILES.items()):
        # Use values from configuration
        gamma_eff = profile.Gamma_eff
        sigma_phi_sq = profile.phase_noise_variance
        
        # Calculate capacity
        capacity_hw = hardware_limited_capacity(snr_linear, gamma_eff, sigma_phi_sq)
        
        # Calculate ceiling
        ceiling = calculate_capacity_ceiling(gamma_eff, sigma_phi_sq)
        
        # Plot capacity curve
        ax.plot(snr_dB, capacity_hw,
                color=profile_colors[i], linewidth=2.5,
                linestyle=profile_styles[i],
                label=f'{name.replace("_", " ")}\n(Γ_eff = {gamma_eff})')
        
        # Add ceiling line
        ax.axhline(y=ceiling, color=profile_colors[i], 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Add ceiling annotation
        ax.annotate(f'Ceiling: {ceiling:.2f} bits/symbol',
                    xy=(45, ceiling), xytext=(35, ceiling + 0.5),
                    arrowprops=dict(arrowstyle='->', color=profile_colors[i], alpha=0.7),
                    fontsize=10, color=profile_colors[i])
    
    # Formatting
    ax.set_xlabel('Nominal SNR ρ [dB]', fontsize=14)
    ax.set_ylabel('Capacity [bits/symbol]', fontsize=14)
    ax.set_title('THz ISL Channel Capacity: Hardware Limitations vs. Ideal AWGN',
                 fontsize=16, pad=15)
    
    # Grid and limits
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(-10, 50)
    ax.set_ylim(0, 12)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, 
              shadow=True, ncol=1, fontsize=11)
    
    # Add text box explaining the phenomenon
    textstr = (
        'Key Insight:\n'
        '• AWGN capacity grows unbounded with power\n'
        '• Hardware impairments create capacity ceilings\n'
        '• Higher Γ_eff → Lower ceiling\n'
        f'• Phase noise: σ_φ² ≈ 0.042 rad² (both profiles)'
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add shaded regions to highlight saturation
    ax.axvspan(30, 50, alpha=0.1, color='gray', label='_nolegend_')
    ax.text(40, 1.0, 'Hardware-limited\nregime', ha='center', fontsize=10,
            style='italic', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('capacity_vs_snr.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('capacity_vs_snr.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical results
    print("\n=== Capacity Ceiling Analysis ===")
    for name, profile in HARDWARE_PROFILES.items():
        gamma_eff = profile.Gamma_eff
        sigma_phi_sq = profile.phase_noise_variance
        
        ceiling = calculate_capacity_ceiling(gamma_eff, sigma_phi_sq)
        
        # Calculate capacity at 30 dB
        cap_at_30dB = hardware_limited_capacity(
            np.array([1000]), gamma_eff, sigma_phi_sq
        )[0]
        
        # Check for valid ceiling before division
        if ceiling > 0:
            saturation_pct = (cap_at_30dB / ceiling) * 100
        else:
            saturation_pct = 0
        
        print(f"\n{name.replace('_', ' ')}:")
        print(f"  Gamma_eff: {gamma_eff:.3f}")
        print(f"  Phase noise variance: {sigma_phi_sq:.4f} rad²")
        print(f"  Capacity ceiling: {ceiling:.3f} bits/symbol")
        print(f"  At 30 dB SNR: {cap_at_30dB:.3f} bits/symbol ({saturation_pct:.1f}% of ceiling)")
        print(f"  Equivalent to: {2**ceiling:.1f}-QAM maximum constellation")

def plot_hardware_components():
    """
    Generate pie charts showing the anatomy of hardware impairment floor.
    As suggested in the review.
    """
    print("\n=== Generating Hardware Components Plot ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Anatomy of the Hardware Impairment Floor (Γ_eff)', fontsize=16, pad=20)
    
    # Custom colors for components
    component_colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red
    
    for ax, (name, profile) in zip(axes, HARDWARE_PROFILES.items()):
        # Get component values
        components = {
            'PA': profile.Gamma_PA,
            'LO': profile.Gamma_LO,
            'ADC': profile.Gamma_ADC
        }
        
        # Calculate percentages
        total = sum(components.values())
        percentages = {k: v/total*100 for k, v in components.items()}
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            components.values(), 
            labels=[f'{k}\n({v:.1e})' for k, v in components.items()],
            autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else f'{pct:.2f}%',
            startangle=90,
            colors=component_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        # Enhance text
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        # Add title and total
        ax.set_title(f'{name.replace("_", " ")}\nΓ_eff = {profile.Gamma_eff:.3f}', 
                    fontsize=12, pad=10)
        
        # Add text showing PA dominance
        if percentages['PA'] > 90:
            ax.text(0, -1.3, f"PA dominates: {percentages['PA']:.1f}%", 
                   ha='center', transform=ax.transAxes, 
                   fontsize=10, style='italic', color=component_colors[0])
    
    plt.tight_layout()
    plt.savefig('hardware_components.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('hardware_components.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print component analysis
    print("\nComponent Analysis:")
    for name, profile in HARDWARE_PROFILES.items():
        print(f"\n{name}:")
        total = profile.Gamma_PA + profile.Gamma_LO + profile.Gamma_ADC
        print(f"  PA contribution: {profile.Gamma_PA/total*100:.1f}%")
        print(f"  LO contribution: {profile.Gamma_LO/total*100:.3f}%")
        print(f"  ADC contribution: {profile.Gamma_ADC/total*100:.2f}%")

def main():
    """Main function to run capacity simulations."""
    print("=== THz ISL ISAC Capacity Simulation ===")
    print(f"Analyzing hardware-limited capacity ceiling effect")
    
    # Verify phase noise calculation
    for name, profile in HARDWARE_PROFILES.items():
        print(f"\n{name} phase noise check:")
        print(f"  Δν = {profile.components.LO_linewidth_Hz/1e3:.0f} kHz")
        print(f"  T = {profile.frame_duration_s*1e6:.0f} μs")
        print(f"  σ_φ² = {profile.phase_noise_variance:.4f} rad²")
    
    # Generate main capacity plot
    simulate_capacity_vs_snr()
    
    # Generate hardware component breakdown
    plot_hardware_components()
    
    print("\n=== Simulation Complete ===")
    print("Output files:")
    print("  - capacity_vs_snr.pdf/png")
    print("  - hardware_components.pdf/png")

if __name__ == "__main__":
    main()