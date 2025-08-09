#!/usr/bin/env python3
"""
conventional_vs_hardware.py - Comparison of conventional vs hardware-aware modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation_config import (
    PhysicalConstants,
    HARDWARE_PROFILES,
    IEEEStyle,
    data_saver
)

# Setup IEEE style
IEEEStyle.setup()

def plot_conventional_vs_hardware(
    hardware_profile='High_Performance',
    f_c=300e9,
    M=64,
    snr_dB_range=None,
    save_name='fig_conventional_vs_hardware'
):
    """
    Plot comparison between conventional AWGN and hardware-aware channel models.
    
    Args:
        hardware_profile: Name of hardware profile to use
        f_c: Carrier frequency [Hz]
        M: Number of pilots/observations
        snr_dB_range: SNR range in dB
        save_name: Output filename
    """
    if snr_dB_range is None:
        snr_dB_range = np.linspace(-10, 50, 61)
    
    # Get hardware parameters
    profile = HARDWARE_PROFILES[hardware_profile]
    sigma_phi2 = profile.phase_noise_variance
    Gamma_eff = profile.Gamma_eff
    
    # Convert SNR to linear
    snr_linear = 10**(snr_dB_range/10)
    
    # Calculate capacities
    C_conventional = np.log2(1 + snr_linear)
    C_hardware = np.log2(1 + (snr_linear * np.exp(-sigma_phi2)) / 
                        (1 + snr_linear * Gamma_eff))
    C_ceiling = np.log2(1 + np.exp(-sigma_phi2) / Gamma_eff)
    
    # Calculate ranging RMSE
    kappa_r = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
    SNR_eff = (snr_linear * np.exp(-sigma_phi2)) / (1 + snr_linear * Gamma_eff)
    
    rmse_conventional = np.sqrt(kappa_r / (M * snr_linear)) * 1000  # mm
    rmse_hardware = np.sqrt(kappa_r * np.exp(sigma_phi2) / (M * SNR_eff)) * 1000  # mm
    
    # Create figure with IEEE style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Capacity comparison
    ax1.plot(snr_dB_range, C_conventional, 
            'b-', linewidth=IEEEStyle.LINE_PROPS['linewidth'],
            label='Conventional (AWGN only)')
    ax1.plot(snr_dB_range, C_hardware, 
            'r-', linewidth=IEEEStyle.LINE_PROPS['linewidth'],
            label='Hardware-aware (This work)')
    ax1.axhline(C_ceiling, color='k', linestyle='--', 
               linewidth=1.5, alpha=0.7,
               label=f'Hardware ceiling ({C_ceiling:.2f} bits/symbol)')
    
    # Add shading to show the gap
    ax1.fill_between(snr_dB_range, C_conventional, C_hardware, 
                     alpha=0.2, color='gray', label='Overestimation gap')
    
    ax1.set_xlabel('Pre-impairment SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_title('(a) Communication Capacity', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax1.grid(True, **IEEEStyle.GRID_PROPS)
    ax1.legend(loc='lower right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax1.set_xlim(-10, 50)
    ax1.set_ylim(0, 8)
    
    # Right: Ranging RMSE comparison
    ax2.semilogy(snr_dB_range, rmse_conventional,
                'b-', linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                label='Conventional CRLB')
    ax2.semilogy(snr_dB_range, rmse_hardware,
                'r-', linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                label='Hardware-aware BCRLB')
    
    # Add hardware floor line
    rmse_floor = np.sqrt(kappa_r * Gamma_eff * np.exp(2*sigma_phi2) / M) * 1000
    ax2.axhline(rmse_floor, color='k', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Hardware floor ({rmse_floor:.3e} mm)')
    
    ax2.set_xlabel('Pre-impairment SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_title('(b) Ranging Accuracy', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax2.grid(True, **IEEEStyle.GRID_PROPS)
    ax2.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax2.set_xlim(-10, 50)
    ax2.set_ylim(1e-4, 1e2)
    
    # Add profile info
    fig.suptitle(f'Conventional vs Hardware-Aware Modeling ({hardware_profile.replace("_", " ")})',
                fontsize=IEEEStyle.FONT_SIZES['title']+2, y=1.02)
    
    # Add parameter box
    param_text = (f'$f_c$ = {f_c/1e9:.0f} GHz\n'
                 f'M = {M} pilots\n'
                 f'$\\Gamma_{{eff}}$ = {Gamma_eff:.3f}\n'
                 f'$\\sigma_\\phi^2$ = {sigma_phi2:.3f}')
    fig.text(0.02, 0.98, param_text,
            transform=fig.transFigure,
            fontsize=IEEEStyle.FONT_SIZES['annotation'],
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    data_to_save = {
        'snr_dB': snr_dB_range.tolist(),
        'C_conventional': C_conventional.tolist(),
        'C_hardware': C_hardware.tolist(),
        'C_ceiling': float(C_ceiling),
        'rmse_conventional_mm': rmse_conventional.tolist(),
        'rmse_hardware_mm': rmse_hardware.tolist(),
        'hardware_profile': hardware_profile,
        'f_c_GHz': f_c/1e9,
        'M_pilots': M
    }
    
    data_saver.save_data(save_name, data_to_save,
                       "Conventional vs hardware-aware modeling comparison")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

if __name__ == "__main__":
    # Generate for different hardware profiles
    for profile in ['State_of_Art', 'High_Performance', 'SWaP_Efficient']:
        plot_conventional_vs_hardware(
            hardware_profile=profile,
            save_name=f'fig_conventional_vs_hardware_{profile.lower()}'
        )