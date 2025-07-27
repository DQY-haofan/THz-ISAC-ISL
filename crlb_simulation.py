#!/usr/bin/env python3
"""
crlb_simulation.py - IEEE Publication Style with Individual Plots
Updated with pointing error model and Monte Carlo averaging
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Tuple, Dict, List
import itertools
import os
from tqdm import tqdm

# Import configuration
from simulation_config import (
    PhysicalConstants, 
    scenario, 
    simulation,
    HARDWARE_PROFILES,
    DerivedParameters,
    ObservableParameters,
    IEEEStyle
)

# Setup IEEE style
IEEEStyle.setup()
colors = IEEEStyle.get_colors()
markers = IEEEStyle.get_markers()
linestyles = IEEEStyle.get_linestyles()

class EnhancedCRLBAnalyzer:
    """Enhanced CRLB analyzer for single ISL with IEEE publication style."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.observable_dim = ObservableParameters.get_observable_dimension()
    
    def calculate_channel_gain(self, distance_m: float, frequency_Hz: float, 
                              antenna_diameter: float = 0.5) -> float:
        """Calculate channel gain magnitude |g|."""
        lambda_c = PhysicalConstants.wavelength(frequency_Hz)
        G_single = scenario.antenna_gain(antenna_diameter, frequency_Hz)
        beta_ch = (lambda_c / (4 * np.pi * distance_m)) * np.sqrt(G_single * G_single)
        return beta_ch
    
    def calculate_bussgang_gain(self, input_backoff_dB: float = 7.0) -> float:
        """Calculate Bussgang gain for PA nonlinearity."""
        kappa = 10 ** (-input_backoff_dB / 10)
        B = 1 - 1.5 * kappa + 1.875 * kappa**2
        return B
    
    def calculate_effective_noise_variance_mc(
        self, SNR_linear: float, channel_gain: float, hardware_profile: str,
        signal_power: float = 1.0, tx_power_dBm: float = 20,
        bandwidth_Hz: float = 10e9, frequency_Hz: float = 300e9,
        antenna_diameter: float = 1.0, n_mc: int = 100
    ) -> Tuple[float, float]:
        """Calculate effective noise variance with Monte Carlo pointing error averaging."""
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        
        P_tx_watts = 10**(tx_power_dBm/10) / 1000
        
        # Monte Carlo loop for pointing error
        pointing_losses = scenario.sample_pointing_loss(
            frequency_Hz, antenna_diameter, n_samples=n_mc
        )
        
        # Average received power with pointing error
        P_rx_avg = P_tx_watts * signal_power * (channel_gain ** 2) * (B ** 2) * np.mean(pointing_losses)
        
        N_0 = P_rx_avg / SNR_linear
        sigma_hw_sq = P_rx_avg * profile.Gamma_eff
        phase_noise_factor = np.exp(profile.phase_noise_variance)
        sigma_DSE_sq = 0.001 * N_0
        
        sigma_eff_sq = N_0 + sigma_hw_sq * phase_noise_factor + sigma_DSE_sq
        
        return sigma_eff_sq, N_0
    
    def calculate_observable_bcrlb_mc(
        self, f_c: float, sigma_eff_sq: float, M: int,
        channel_gain: float, B: float, sigma_phi_sq: float,
        T_CPI: float = 1e-3, signal_power: float = 1.0,
        antenna_diameter: float = 1.0, n_mc: int = 100
    ) -> Dict[str, float]:
        """Calculate BCRLB for observable parameters with Monte Carlo pointing error."""
        phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
        doppler_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2 * T_CPI**2)
        
        # Monte Carlo averaging
        pointing_losses = scenario.sample_pointing_loss(f_c, antenna_diameter, n_samples=n_mc)
        
        # Average performance
        P_rx_base = signal_power * (channel_gain**2) * (B**2)
        noise_terms = sigma_eff_sq / (M * P_rx_base * pointing_losses)
        phase_penalty = np.exp(sigma_phi_sq)
        
        bcrlb_range = phase_term * np.mean(noise_terms) * phase_penalty
        bcrlb_range_rate = doppler_term * np.mean(noise_terms) * phase_penalty
        
        return {
            'range': bcrlb_range,
            'range_rate': bcrlb_range_rate
        }
    
    # =========================================================================
    # INDIVIDUAL PLOT FUNCTIONS
    # =========================================================================
    
    def plot_ranging_crlb_vs_snr(self, save_name='fig_ranging_crlb_vs_snr'):
        """Plot ranging CRLB vs SNR for all hardware profiles - IEEE style."""
        print(f"\n=== Generating {save_name} ===")
        
        # Create figure with IEEE single column size
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
        
        # Parameters
        frequency_Hz = 300e9
        antenna_diameter = 1.0
        tx_power_dBm = 30
        
        # All hardware profiles
        profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
        
        # Calculate for each profile
        for i, hardware_profile in enumerate(profiles_to_plot):
            if hardware_profile not in HARDWARE_PROFILES:
                continue
                
            profile = HARDWARE_PROFILES[hardware_profile]
            B = self.calculate_bussgang_gain()
            
            ranging_rmse_mm = []
            
            # Progress bar for Monte Carlo
            print(f"  Processing {hardware_profile}...")
            
            for snr_dB in tqdm(simulation.SNR_dB_array, desc=f"    SNR sweep", leave=False):
                snr_linear = 10 ** (snr_dB / 10)
                
                g = self.calculate_channel_gain(scenario.R_default, frequency_Hz, antenna_diameter)
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance_mc(
                    snr_linear, g, hardware_profile, tx_power_dBm=tx_power_dBm,
                    frequency_Hz=frequency_Hz, antenna_diameter=antenna_diameter,
                    n_mc=100  # Reduced for speed
                )
                
                bcrlbs = self.calculate_observable_bcrlb_mc(
                    frequency_Hz, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance,
                    antenna_diameter=antenna_diameter, n_mc=100
                )
                
                rmse_m = np.sqrt(bcrlbs['range'])
                ranging_rmse_mm.append(rmse_m * 1000)
            
            # Plot with IEEE style
            ax.semilogy(simulation.SNR_dB_array, ranging_rmse_mm,
                       color=colors[i], 
                       linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                       marker=markers[i], 
                       markersize=IEEEStyle.LINE_PROPS['markersize'],
                       markevery=10,
                       markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                       markerfacecolor='white',
                       label=f'{hardware_profile.replace("_", " ")}')
        
        # Add performance thresholds
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(35, 1.2, '1 mm', fontsize=IEEEStyle.FONT_SIZES['annotation'], 
                ha='center', color='gray')
        
        ax.axhline(y=0.1, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(35, 0.12, '0.1 mm', fontsize=IEEEStyle.FONT_SIZES['annotation'], 
                ha='center', color='green')
        
        # Labels and formatting
        ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('Ranging Performance vs. SNR (All Hardware Profiles)', 
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        
        # Grid
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        
        # Legend
        ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'],
                 frameon=True, edgecolor='black', framealpha=0.9)
        
        # Axis limits
        ax.set_xlim(-10, 50)
        ax.set_ylim(1e-2, 1e3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save to results folder
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")
    
    def plot_hardware_comparison(self, save_name='fig_hardware_comparison'):
        """Plot hardware profile comparison - IEEE style."""
        print(f"\n=== Generating {save_name} ===")
        
        # Create figure with IEEE double column size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=IEEEStyle.FIG_SIZES['double'])
        
        # Parameters
        snr_dB_values = [10, 20, 30]
        f_c_GHz = 300
        f_c_Hz = f_c_GHz * 1e9
        antenna_diameter = 1.0
        tx_power_dBm = 30
        
        profiles = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
        
        # Data storage
        ranging_data = {profile: [] for profile in profiles}
        velocity_data = {profile: [] for profile in profiles}
        
        for profile_name in profiles:
            if profile_name not in HARDWARE_PROFILES:
                continue
                
            profile = HARDWARE_PROFILES[profile_name]
            
            g = self.calculate_channel_gain(scenario.R_default, f_c_Hz, antenna_diameter)
            B = self.calculate_bussgang_gain()
            
            for snr_dB in snr_dB_values:
                snr_linear = 10**(snr_dB/10)
                
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance_mc(
                    snr_linear, g, profile_name, tx_power_dBm=tx_power_dBm,
                    frequency_Hz=f_c_Hz, antenna_diameter=antenna_diameter, n_mc=100
                )
                
                bcrlbs = self.calculate_observable_bcrlb_mc(
                    f_c_Hz, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance,
                    antenna_diameter=antenna_diameter, n_mc=100
                )
                
                ranging_rmse_m = np.sqrt(bcrlbs['range'])
                ranging_data[profile_name].append(ranging_rmse_m * 1000)
                
                velocity_rmse_ms = np.sqrt(bcrlbs['range_rate'])
                velocity_data[profile_name].append(velocity_rmse_ms)
        
        # Plot ranging RMSE (without value labels)
        x = np.arange(len(snr_dB_values))
        width = 0.2
        
        for i, profile_name in enumerate(profiles):
            if profile_name not in ranging_data:
                continue
            offset = (i - 1.5) * width
            bars = ax1.bar(x + offset, ranging_data[profile_name], width,
                           label=profile_name.replace('_', ' '),
                           color=colors[i], alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax1.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax1.set_title('(a) Ranging Performance', fontsize=IEEEStyle.FONT_SIZES['title'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(snr_dB_values)
        ax1.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1, loc='upper right')
        ax1.grid(True, axis='y', **IEEEStyle.GRID_PROPS)
        ax1.set_yscale('log')
        
        # Plot velocity RMSE (without value labels)
        for i, profile_name in enumerate(profiles):
            if profile_name not in velocity_data:
                continue
            offset = (i - 1.5) * width
            bars = ax2.bar(x + offset, velocity_data[profile_name], width,
                           label=profile_name.replace('_', ' '),
                           color=colors[i], alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax2.set_ylabel('Velocity RMSE (m/s)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax2.set_title('(b) Velocity Estimation', fontsize=IEEEStyle.FONT_SIZES['title'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(snr_dB_values)
        ax2.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1, loc='upper right')
        ax2.grid(True, axis='y', **IEEEStyle.GRID_PROPS)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")
    
    def plot_pointing_error_sensitivity(self, save_name='fig_pointing_error_sensitivity'):
        """Plot sensitivity to pointing error - NEW function."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
        
        # Parameters
        f_c = 300e9
        antenna_diameter = 1.0
        tx_power_dBm = 30
        hardware_profile = "High_Performance"
        pointing_errors_urad = [0.5, 1.0, 2.0]  # µrad
        
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        g = self.calculate_channel_gain(scenario.R_default, f_c, antenna_diameter)
        
        for idx, pe_urad in enumerate(pointing_errors_urad):
            pe_rad = pe_urad * 1e-6
            ranging_rmse_mm = []
            
            # Override scenario pointing error temporarily
            original_pe = scenario.pointing_error_rms_rad
            scenario.pointing_error_rms_rad = pe_rad
            
            for snr_dB in simulation.SNR_dB_array:
                snr_linear = 10**(snr_dB/10)
                
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance_mc(
                    snr_linear, g, hardware_profile, tx_power_dBm=tx_power_dBm,
                    frequency_Hz=f_c, antenna_diameter=antenna_diameter, n_mc=100
                )
                
                bcrlbs = self.calculate_observable_bcrlb_mc(
                    f_c, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance,
                    antenna_diameter=antenna_diameter, n_mc=100
                )
                
                ranging_rmse_mm.append(np.sqrt(bcrlbs['range']) * 1000)
            
            # Restore original
            scenario.pointing_error_rms_rad = original_pe
            
            # Plot
            ax.semilogy(simulation.SNR_dB_array, ranging_rmse_mm,
                       color=colors[idx], 
                       linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                       linestyle=linestyles[idx],
                       marker=markers[idx],
                       markersize=IEEEStyle.LINE_PROPS['markersize'],
                       markevery=10,
                       markerfacecolor='white',
                       markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                       label=f'σ_θ = {pe_urad} µrad')
        
        # Add performance threshold
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(35, 1.2, 'Sub-mm threshold', 
               fontsize=IEEEStyle.FONT_SIZES['annotation'], color='gray')
        
        # Labels
        ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('Impact of Pointing Error on Ranging Performance\n(High Performance Hardware)',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
        ax.set_xlim(-10, 50)
        ax.set_ylim(1e-2, 1e3)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")
    
    def plot_feasibility_map(self, save_name='fig_feasibility_map'):
        """Plot 2D feasibility map for antenna size vs transmit power - NEW."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['square'])
        
        # Parameter ranges
        antenna_diameters = np.linspace(0.3, 2.0, 20)
        tx_powers_dBm = np.linspace(10, 33, 20)
        
        # Fixed parameters
        f_c = 300e9
        distance = scenario.R_default
        hardware_profile = "High_Performance"
        
        # Thresholds
        min_link_margin_dB = 3  # Minimum link margin for closure
        max_ranging_rmse_mm = 1.0  # Sub-mm requirement
        min_capacity_bits = 2.0  # Good communication
        
        # Create meshgrid
        D, P = np.meshgrid(antenna_diameters, tx_powers_dBm)
        
        # Initialize feasibility map
        feasibility = np.zeros_like(D)
        
        print("  Computing feasibility map...")
        for i in tqdm(range(D.shape[0]), desc="    Power levels"):
            for j in range(D.shape[1]):
                ant_diam = D[i,j]
                tx_power = P[i,j]
                
                # Check link closure
                ant_gain = scenario.antenna_gain_dB(ant_diam, f_c)
                budget = DerivedParameters.link_budget_dB(
                    tx_power, ant_gain, ant_gain, distance, f_c
                )
                noise_dBm = DerivedParameters.thermal_noise_power_dBm(10e9, noise_figure_dB=8)
                link_margin = budget['rx_power_dBm'] - noise_dBm
                
                if link_margin < min_link_margin_dB:
                    feasibility[i,j] = 0  # Link doesn't close
                    continue
                
                # Check ranging performance at SNR = 20 dB
                snr_linear = 100
                g = self.calculate_channel_gain(distance, f_c, ant_diam)
                B = self.calculate_bussgang_gain()
                profile = HARDWARE_PROFILES[hardware_profile]
                
                sigma_eff_sq, _ = self.calculate_effective_noise_variance_mc(
                    snr_linear, g, hardware_profile, tx_power_dBm=tx_power,
                    frequency_Hz=f_c, antenna_diameter=ant_diam, n_mc=50
                )
                
                bcrlbs = self.calculate_observable_bcrlb_mc(
                    f_c, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance,
                    antenna_diameter=ant_diam, n_mc=50
                )
                
                ranging_rmse_mm = np.sqrt(bcrlbs['range']) * 1000
                
                # Simplified capacity estimate
                capacity = np.log2(1 + snr_linear / (1 + snr_linear * profile.Gamma_eff))
                
                # Determine feasibility
                comm_ok = capacity >= min_capacity_bits
                sense_ok = ranging_rmse_mm <= max_ranging_rmse_mm
                
                if comm_ok and sense_ok:
                    feasibility[i,j] = 3  # Both OK
                elif comm_ok:
                    feasibility[i,j] = 1  # Communication only
                elif sense_ok:
                    feasibility[i,j] = 2  # Sensing only
                else:
                    feasibility[i,j] = 0.5  # Link OK but neither meets specs
        
        # Create custom colormap
        cmap = plt.cm.colors.ListedColormap(['darkred', 'orange', 'lightcoral', 'lightblue', 'darkgreen'])
        bounds = [0, 0.25, 0.75, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        im = ax.contourf(D, P, feasibility, levels=bounds, cmap=cmap, norm=norm)
        
        # Add contour lines
        ax.contour(D, P, feasibility, levels=[0.5, 1.5, 2.5], colors='black', 
                   linewidths=1, alpha=0.5)
        
        # Labels
        ax.set_xlabel('Antenna Diameter (m)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Transmit Power (dBm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title(f'ISAC Feasibility Map at {distance/1e3:.0f} km\n' + 
                    '(C ≥ 2 bits/symbol, RMSE ≤ 1 mm)',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', edgecolor='black', label='Link Fails'),
            Patch(facecolor='orange', edgecolor='black', label='Link OK, Neither Meets Spec'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Communication Only'),
            Patch(facecolor='lightblue', edgecolor='black', label='Sensing Only'),
            Patch(facecolor='darkgreen', edgecolor='black', label='ISAC Feasible')
        ]
        ax.legend(handles=legend_elements, loc='lower right', 
                 fontsize=IEEEStyle.FONT_SIZES['legend']-1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Mark recommended region
        from matplotlib.patches import Rectangle
        rect = Rectangle((0.8, 28), 1.2, 5, fill=False, 
                        edgecolor='white', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.text(1.4, 30.5, 'Recommended', ha='center', color='white',
               fontsize=IEEEStyle.FONT_SIZES['annotation'], 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def main():
    """Main function to generate all CRLB analysis plots."""
    print("=== THz ISL ISAC CRLB Analysis (IEEE Style) ===")
    print("With pointing error Monte Carlo averaging")
    print(f"Current font sizes: {IEEEStyle.FONT_SIZES}")
    
    # Print observability warning once
    ObservableParameters.print_observability_warning()
    
    analyzer = EnhancedCRLBAnalyzer()
    
    # Generate all plots
    analyzer.plot_ranging_crlb_vs_snr()
    analyzer.plot_hardware_comparison()
    analyzer.plot_pointing_error_sensitivity()  # NEW
    analyzer.plot_feasibility_map()  # NEW
    
    print("\n=== CRLB Analysis Complete ===")
    print("Generated files in results/:")
    print("- fig_ranging_crlb_vs_snr.pdf/png")
    print("- fig_hardware_comparison.pdf/png")
    print("- fig_pointing_error_sensitivity.pdf/png")
    print("- fig_feasibility_map.pdf/png")

if __name__ == "__main__":
    main()