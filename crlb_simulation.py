#!/usr/bin/env python3
"""
crlb_simulation.py - IEEE Publication Style with Individual Plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Tuple, Dict, List
import itertools

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
    
    def calculate_effective_noise_variance(
        self, SNR_linear: float, channel_gain: float, hardware_profile: str,
        signal_power: float = 1.0, tx_power_dBm: float = 20,
        bandwidth_Hz: float = 10e9
    ) -> Tuple[float, float]:
        """Calculate effective noise variance including hardware impairments."""
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        
        P_tx_watts = 10**(tx_power_dBm/10) / 1000
        P_rx = P_tx_watts * signal_power * (channel_gain ** 2) * (B ** 2)
        
        N_0 = P_rx / SNR_linear
        sigma_hw_sq = P_rx * profile.Gamma_eff
        phase_noise_factor = np.exp(profile.phase_noise_variance)
        sigma_DSE_sq = 0.001 * N_0
        
        sigma_eff_sq = N_0 + sigma_hw_sq * phase_noise_factor + sigma_DSE_sq
        
        return sigma_eff_sq, N_0
    
    def calculate_observable_bcrlb(
        self, f_c: float, sigma_eff_sq: float, M: int,
        channel_gain: float, B: float, sigma_phi_sq: float,
        T_CPI: float = 1e-3, signal_power: float = 1.0
    ) -> Dict[str, float]:
        """Calculate BCRLB for observable parameters only."""
        phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
        doppler_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2 * T_CPI**2)
        
        P_rx = signal_power * (channel_gain**2) * (B**2)
        noise_term = sigma_eff_sq / (M * P_rx)
        phase_penalty = np.exp(sigma_phi_sq)
        
        bcrlb_range = phase_term * noise_term * phase_penalty
        bcrlb_range_rate = doppler_term * noise_term * phase_penalty
        
        return {
            'range': bcrlb_range,
            'range_rate': bcrlb_range_rate
        }
    
    # =========================================================================
    # INDIVIDUAL PLOT FUNCTIONS
    # =========================================================================
    
    def plot_ranging_crlb_vs_snr(self, save_name='fig_ranging_crlb_vs_snr'):
        """Plot ranging CRLB vs SNR for different frequencies - IEEE style."""
        print(f"\n=== Generating {save_name} ===")
        
        # Create figure with IEEE single column size
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
        
        # Parameters
        frequencies_GHz = [100, 300, 600]
        frequencies_Hz = [f * 1e9 for f in frequencies_GHz]
        hardware_profile = "High_Performance"
        antenna_diameter = 1.0
        tx_power_dBm = 30
        
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        
        # Calculate for each frequency
        for i, (f_GHz, f_Hz) in enumerate(zip(frequencies_GHz, frequencies_Hz)):
            ranging_rmse_mm = []
            
            for snr_dB in simulation.SNR_dB_array:
                snr_linear = 10 ** (snr_dB / 10)
                
                g = self.calculate_channel_gain(scenario.R_default, f_Hz, antenna_diameter)
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance(
                    snr_linear, g, hardware_profile, tx_power_dBm=tx_power_dBm
                )
                
                bcrlbs = self.calculate_observable_bcrlb(
                    f_Hz, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance
                )
                
                rmse_m = np.sqrt(bcrlbs['range'])
                ranging_rmse_mm.append(rmse_m * 1000)
            
            # Plot with IEEE style
            ax.semilogy(simulation.SNR_dB_array, ranging_rmse_mm,
                       color=colors[i], 
                       linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                       marker=markers[i], 
                       markersize=IEEEStyle.LINE_PROPS['markersize'],
                       markevery=8,
                       markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                       markerfacecolor='white',
                       label=f'{f_GHz} GHz')
        
        # Add performance thresholds
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(35, 1.2, '1 mm', fontsize=IEEEStyle.FONT_SIZES['annotation'], 
                ha='center', color='gray')
        
        # Labels and formatting
        ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('Ranging Performance vs. SNR', fontsize=IEEEStyle.FONT_SIZES['title'])
        
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
        
        # Save both PDF and PNG
        plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}.pdf and {save_name}.png")
    
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
        
        profiles = ["High_Performance", "SWaP_Efficient", "Low_Cost"]
        
        # Data storage
        ranging_data = {profile: [] for profile in profiles}
        velocity_data = {profile: [] for profile in profiles}
        
        for profile_name in profiles:
            profile = HARDWARE_PROFILES[profile_name]
            
            g = self.calculate_channel_gain(scenario.R_default, f_c_Hz, antenna_diameter)
            B = self.calculate_bussgang_gain()
            
            for snr_dB in snr_dB_values:
                snr_linear = 10**(snr_dB/10)
                
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance(
                    snr_linear, g, profile_name, tx_power_dBm=tx_power_dBm
                )
                
                bcrlbs = self.calculate_observable_bcrlb(
                    f_c_Hz, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance
                )
                
                ranging_rmse_m = np.sqrt(bcrlbs['range'])
                ranging_data[profile_name].append(ranging_rmse_m * 1000)
                
                velocity_rmse_ms = np.sqrt(bcrlbs['range_rate'])
                velocity_data[profile_name].append(velocity_rmse_ms)
        
        # Plot ranging RMSE
        x = np.arange(len(snr_dB_values))
        width = 0.25
        
        for i, profile_name in enumerate(profiles):
            offset = (i - 1) * width
            bars = ax1.bar(x + offset, ranging_data[profile_name], width,
                           label=profile_name.replace('_', ' '),
                           color=colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', 
                        fontsize=IEEEStyle.FONT_SIZES['annotation']-1)
        
        ax1.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax1.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax1.set_title('(a) Ranging Performance', fontsize=IEEEStyle.FONT_SIZES['title'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(snr_dB_values)
        ax1.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
        ax1.grid(True, axis='y', **IEEEStyle.GRID_PROPS)
        ax1.set_yscale('log')
        
        # Plot velocity RMSE
        for i, profile_name in enumerate(profiles):
            offset = (i - 1) * width
            bars = ax2.bar(x + offset, velocity_data[profile_name], width,
                           label=profile_name.replace('_', ' '),
                           color=colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontsize=IEEEStyle.FONT_SIZES['annotation']-1)
        
        ax2.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax2.set_ylabel('Velocity RMSE (m/s)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax2.set_title('(b) Velocity Estimation', fontsize=IEEEStyle.FONT_SIZES['title'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(snr_dB_values)
        ax2.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
        ax2.grid(True, axis='y', **IEEEStyle.GRID_PROPS)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}.pdf and {save_name}.png")
    
    def plot_hardware_parameter_scan(self, save_name='fig_hardware_scan'):
        """Plot performance vs hardware quality factor - IEEE style."""
        print(f"\n=== Generating {save_name} ===")
        
        # Create figure
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
        
        # Parameters
        f_c = 300e9
        antenna_diameter = 1.0
        tx_power_dBm = 30
        gamma_eff_range = np.logspace(-3, -1, 30)
        snr_levels_dB = [10, 20, 30, 40]
        
        for idx, snr_dB in enumerate(snr_levels_dB):
            snr_linear = 10**(snr_dB/10)
            ranging_rmse_mm = []
            
            for gamma_eff in gamma_eff_range:
                temp_profile = HARDWARE_PROFILES["Custom"]
                original_gamma = temp_profile.Gamma_eff
                temp_profile.Gamma_eff = gamma_eff
                
                g = self.calculate_channel_gain(scenario.R_default, f_c, antenna_diameter)
                B = self.calculate_bussgang_gain()
                
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance(
                    snr_linear, g, "Custom", tx_power_dBm=tx_power_dBm
                )
                
                bcrlbs = self.calculate_observable_bcrlb(
                    f_c, sigma_eff_sq, simulation.n_pilots,
                    g, B, temp_profile.phase_noise_variance
                )
                
                ranging_rmse_mm.append(np.sqrt(bcrlbs['range']) * 1000)
                temp_profile.Gamma_eff = original_gamma
            
            # Plot
            ax.loglog(gamma_eff_range, ranging_rmse_mm,
                     color=colors[idx], 
                     linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                     linestyle=linestyles[idx],
                     marker=markers[idx],
                     markersize=IEEEStyle.LINE_PROPS['markersize'],
                     markevery=6,
                     markerfacecolor='white',
                     markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                     label=f'SNR = {snr_dB} dB')
        
        # Mark current hardware profiles
        for name, profile in HARDWARE_PROFILES.items():
            if name not in ["Custom", "State_of_Art"]:
                ax.axvline(x=profile.Gamma_eff, color='gray', 
                          linestyle=':', alpha=0.5, linewidth=1.2)
                ax.text(profile.Gamma_eff*1.1, 0.05, name.split('_')[0],
                       rotation=90, fontsize=IEEEStyle.FONT_SIZES['annotation']-1,
                       alpha=0.7, va='bottom')
        
        # Performance threshold
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(2e-3, 1.2, 'Sub-mm threshold', 
               fontsize=IEEEStyle.FONT_SIZES['annotation'], color='green')
        
        # Labels
        ax.set_xlabel('Hardware Quality Factor $\Gamma_{eff}$', 
                     fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('Impact of Hardware Quality on Ranging Performance',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        ax.legend(loc='upper left', fontsize=IEEEStyle.FONT_SIZES['legend'])
        ax.set_xlim(1e-3, 1e-1)
        ax.set_ylim(1e-2, 1e2)
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}.pdf and {save_name}.png")
    
    def plot_operational_regions(self, save_name='fig_operational_regions'):
        """Plot 2D operational regions map - IEEE style."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
        
        # Parameter ranges
        snr_dB_range = np.linspace(-10, 50, 40)
        gamma_eff_range = np.logspace(-3, -1, 40)
        
        SNR, GAMMA = np.meshgrid(snr_dB_range, gamma_eff_range)
        regions = np.zeros_like(SNR)
        
        # Calculate regions
        for i in range(SNR.shape[0]):
            for j in range(SNR.shape[1]):
                gamma_eff = GAMMA[i,j]
                snr_hw_limit = DerivedParameters.find_snr_for_hardware_limit(gamma_eff, 0.95)
                
                if SNR[i,j] < snr_hw_limit - 10:
                    regions[i,j] = 1  # Power-limited
                elif SNR[i,j] > snr_hw_limit:
                    regions[i,j] = 3  # Hardware-limited
                else:
                    regions[i,j] = 2  # Transition
        
        # Create custom colormap
        cmap = plt.cm.colors.ListedColormap(['white', '#3498db', '#f1c40f', '#e74c3c'])
        bounds = [0, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot regions
        im = ax.contourf(SNR, GAMMA, regions, levels=bounds, cmap=cmap, norm=norm)
        
        # Add hardware profile lines
        for name, profile in HARDWARE_PROFILES.items():
            if name != "Custom":
                ax.axhline(y=profile.Gamma_eff, color='black', linewidth=2)
                ax.text(45, profile.Gamma_eff*1.2, name.replace('_', ' '),
                       fontsize=IEEEStyle.FONT_SIZES['annotation'], ha='center')
        
        # Add iso-capacity lines
        snr_vals = np.linspace(0, 50, 100)
        for target_cap in [2, 4]:
            gamma_vals = 1 / (10**(snr_vals/10) * (2**target_cap - 1))
            mask = (gamma_vals >= 1e-3) & (gamma_vals <= 1e-1)
            ax.plot(snr_vals[mask], gamma_vals[mask], 'k--', alpha=0.3, linewidth=1)
        
        # Labels
        ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Hardware Quality Factor $\Gamma_{eff}$', 
                     fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_yscale('log')
        ax.set_title('Operational Regions for THz ISL ISAC',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        
        # Add text labels
        ax.text(5, 0.05, 'Power-Limited', ha='center', 
               fontsize=IEEEStyle.FONT_SIZES['annotation'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.7))
        ax.text(40, 0.005, 'Hardware-Limited', ha='center',
               fontsize=IEEEStyle.FONT_SIZES['annotation'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}.pdf and {save_name}.png")
    
    def plot_frequency_scaling(self, save_name='fig_frequency_scaling'):
        """Plot frequency scaling analysis - IEEE style."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
        
        # Frequency range
        frequencies_GHz = np.array([100, 200, 300, 400, 500, 600, 800, 1000])
        frequencies_Hz = frequencies_GHz * 1e9
        
        # Fixed parameters
        snr_dB = 20
        snr_linear = 10**(snr_dB/10)
        hardware_profile = "High_Performance"
        antenna_diameter = 1.0
        
        ranging_rmse_mm = []
        link_margin_dB = []
        
        for f_Hz in frequencies_Hz:
            g = self.calculate_channel_gain(scenario.R_default, f_Hz, antenna_diameter)
            B = self.calculate_bussgang_gain()
            
            # Calculate link margin
            ant_gain = scenario.antenna_gain_dB(antenna_diameter, f_Hz)
            budget = DerivedParameters.link_budget_dB(
                30, ant_gain, ant_gain, scenario.R_default, f_Hz
            )
            noise_dBm = DerivedParameters.thermal_noise_power_dBm(10e9, noise_figure_dB=8)
            margin = budget['rx_power_dBm'] - noise_dBm
            link_margin_dB.append(margin)
            
            if margin < 0:
                ranging_rmse_mm.append(np.nan)
                continue
            
            profile = HARDWARE_PROFILES[hardware_profile]
            sigma_eff_sq, N_0 = self.calculate_effective_noise_variance(
                snr_linear, g, hardware_profile, tx_power_dBm=30
            )
            
            bcrlbs = self.calculate_observable_bcrlb(
                f_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            
            ranging_rmse_mm.append(np.sqrt(bcrlbs['range']) * 1000)
        
        # Create twin axis
        ax2 = ax.twinx()
        
        # Plot ranging performance
        valid_mask = ~np.isnan(ranging_rmse_mm)
        line1 = ax.loglog(frequencies_GHz[valid_mask], 
                         np.array(ranging_rmse_mm)[valid_mask],
                         color=colors[0], 
                         linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                         marker=markers[0], 
                         markersize=IEEEStyle.LINE_PROPS['markersize'],
                         markerfacecolor='white',
                         markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                         label='Ranging RMSE')
        
        # Add theoretical scaling
        f_ref = 300
        rmse_ref = np.array(ranging_rmse_mm)[frequencies_GHz == f_ref][0]
        theoretical = rmse_ref * (f_ref / frequencies_GHz)**2
        line2 = ax.loglog(frequencies_GHz, theoretical, 
                         color=colors[0], linestyle='--', 
                         linewidth=IEEEStyle.LINE_PROPS['linewidth']-0.5,
                         label='$f^{-2}$ scaling')
        
        # Plot link margin
        line3 = ax2.plot(frequencies_GHz, link_margin_dB, 
                        color=colors[1], 
                        linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                        marker=markers[1], 
                        markersize=IEEEStyle.LINE_PROPS['markersize'],
                        markerfacecolor='white',
                        markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                        label='Link Margin')
        
        # Add zero margin line
        ax2.axhline(y=0, color=colors[1], linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Mark invalid regions
        for i, (f, valid) in enumerate(zip(frequencies_GHz, ~np.isnan(ranging_rmse_mm))):
            if not valid:
                ax.axvspan(f*0.9, f*1.1, alpha=0.2, color='red')
        
        # Labels
        ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'], 
                     color=colors[0])
        ax2.set_ylabel('Link Margin (dB)', fontsize=IEEEStyle.FONT_SIZES['label'], 
                      color=colors[1])
        ax.set_title('Frequency Scaling of Ranging Performance',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        ax.set_xlim(90, 1100)
        
        # Color the axes
        ax.tick_params(axis='y', labelcolor=colors[0])
        ax2.tick_params(axis='y', labelcolor=colors[1])
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', 
                 fontsize=IEEEStyle.FONT_SIZES['legend'])
        
        plt.tight_layout()
        plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_name}.pdf and {save_name}.png")

def main():
    """Main function to generate all CRLB analysis plots."""
    print("=== THz ISL ISAC CRLB Analysis (IEEE Style) ===")
    print("\nGenerating individual plots with adjustable font sizes...")
    print(f"Current font sizes: {IEEEStyle.FONT_SIZES}")
    
    # Print observability warning once
    ObservableParameters.print_observability_warning()
    
    analyzer = EnhancedCRLBAnalyzer()
    
    # Generate all individual plots
    analyzer.plot_ranging_crlb_vs_snr()
    analyzer.plot_hardware_comparison()
    analyzer.plot_hardware_parameter_scan()
    analyzer.plot_operational_regions()
    analyzer.plot_frequency_scaling()
    
    print("\n=== CRLB Analysis Complete ===")
    print("Generated files:")
    print("- fig_ranging_crlb_vs_snr.pdf/png")
    print("- fig_hardware_comparison.pdf/png")
    print("- fig_hardware_scan.pdf/png")
    print("- fig_operational_regions.pdf/png")
    print("- fig_frequency_scaling.pdf/png")

if __name__ == "__main__":
    main()