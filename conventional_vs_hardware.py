# 文件: conventional_vs_hardware.py (完整替换)
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

def plot_conventional_vs_hardware_all(
    f_c=300e9,
    M=64,
    snr_dB_range=None,
    save_name='fig_conventional_vs_hardware'
):
    """
    Plot comparison between conventional and hardware-aware models for all profiles.
    """
    if snr_dB_range is None:
        snr_dB_range = np.linspace(-10, 50, 61)
    
    # Convert SNR to linear
    snr_linear = 10**(snr_dB_range/10)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Colors and styles for different hardware profiles
    profiles_to_plot = ['State_of_Art', 'High_Performance', 'Low_Cost']
    colors = ['blue', 'orange', 'red']
    linestyles_hw = ['-', '-', '-']  # Solid for hardware-aware
    linestyles_conv = ['--', '--', '--']  # Dashed for conventional
    
    # Data storage
    data_to_save = {
        'snr_dB': snr_dB_range.tolist(),
        'f_c_GHz': f_c/1e9,
        'M_pilots': M,
        'profiles': profiles_to_plot
    }
    
    # Conventional capacity (same for all profiles)
    C_conventional = np.log2(1 + snr_linear)
    
    # Plot for each hardware profile
    for idx, profile_name in enumerate(profiles_to_plot):
        profile = HARDWARE_PROFILES[profile_name]
        sigma_phi2 = profile.phase_noise_variance
        Gamma_eff = profile.Gamma_eff
        
        # Hardware-aware capacity
        C_hardware = np.log2(1 + (snr_linear * np.exp(-sigma_phi2)) / 
                            (1 + snr_linear * Gamma_eff))
        C_ceiling = np.log2(1 + np.exp(-sigma_phi2) / Gamma_eff)
        
        # Ranging RMSE
        kappa_r = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
        SNR_eff = (snr_linear * np.exp(-sigma_phi2)) / (1 + snr_linear * Gamma_eff)
        rmse_hardware = np.sqrt(kappa_r * np.exp(sigma_phi2) / (M * SNR_eff)) * 1000  # mm
        
        # Store data
        data_to_save[f'C_hardware_{profile_name}'] = C_hardware.tolist()
        data_to_save[f'C_ceiling_{profile_name}'] = float(C_ceiling)
        data_to_save[f'rmse_hardware_mm_{profile_name}'] = rmse_hardware.tolist()
        
        # Plot capacity (left)
        if idx == 0:  # Only plot conventional once
            ax1.plot(snr_dB_range, C_conventional, 
                    'k--', linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                    alpha=0.6, label='Conventional (AWGN)')
        
        ax1.plot(snr_dB_range, C_hardware, 
                color=colors[idx], linestyle=linestyles_hw[idx],
                linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                label=f'{profile_name.replace("_", " ")}')
        
        # Add ceiling as thin dotted line
        ax1.axhline(C_ceiling, color=colors[idx], linestyle=':', 
                   linewidth=1.0, alpha=0.5)
        
        # Plot ranging RMSE (right)
        if idx == 0:  # Only plot conventional once
            rmse_conventional = np.sqrt(kappa_r / (M * snr_linear)) * 1000
            ax2.semilogy(snr_dB_range, rmse_conventional,
                        'k--', linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                        alpha=0.6, label='Conventional CRLB')
        
        ax2.semilogy(snr_dB_range, rmse_hardware,
                    color=colors[idx], linestyle=linestyles_hw[idx],
                    linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                    label=f'{profile_name.replace("_", " ")}')
    
    # Configure left subplot (Capacity)
    ax1.set_xlabel('Pre-impairment SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_title('(a) Communication Capacity', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax1.grid(True, **IEEEStyle.GRID_PROPS)
    ax1.legend(loc='lower right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax1.set_xlim(-10, 50)
    ax1.set_ylim(0, 8)
    
    # Configure right subplot (Ranging)
    ax2.set_xlabel('Pre-impairment SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_title('(b) Ranging Accuracy', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax2.grid(True, **IEEEStyle.GRID_PROPS)
    ax2.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax2.set_xlim(-10, 50)
    ax2.set_ylim(1e-4, 1e2)
    
    # Main title
    fig.suptitle('Conventional vs Hardware-Aware Modeling: Impact of Hardware Quality',
                fontsize=IEEEStyle.FONT_SIZES['title']+1, y=1.02)
    
    # Add parameter info box
    param_text = f'$f_c$ = {f_c/1e9:.0f} GHz, M = {M} pilots'
    fig.text(0.5, 0.02, param_text,
            ha='center', va='bottom',
            fontsize=IEEEStyle.FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Conventional vs hardware-aware modeling comparison (all profiles)")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

if __name__ == "__main__":
    plot_conventional_vs_hardware_all()