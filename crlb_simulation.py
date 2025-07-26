#!/usr/bin/env python3
"""
crlb_simulation.py - Enhanced version with combined visualizations
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
    ObservableParameters
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
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
})

colors = sns.color_palette("husl", 6)

class EnhancedCRLBAnalyzer:
    """Enhanced CRLB analyzer for single ISL with proper observability handling."""
    
    def __init__(self):
        """Initialize analyzer and print observability warning."""
        ObservableParameters.print_observability_warning()
        self.observable_dim = ObservableParameters.get_observable_dimension()
        print(f"CRLB analysis will be performed for {self.observable_dim} observable parameters\n")
    
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
        # Phase sensitivity for range
        phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
        
        # Doppler sensitivity for range-rate
        doppler_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2 * T_CPI**2)
        
        # Common terms
        P_rx = signal_power * (channel_gain**2) * (B**2)
        noise_term = sigma_eff_sq / (M * P_rx)
        phase_penalty = np.exp(sigma_phi_sq)
        
        # BCRLBs for observable parameters
        bcrlb_range = phase_term * noise_term * phase_penalty
        bcrlb_range_rate = doppler_term * noise_term * phase_penalty
        
        return {
            'range': bcrlb_range,
            'range_rate': bcrlb_range_rate
        }
    
    def find_operational_regions(self, snr_dB_array: np.ndarray, 
                                gamma_eff_array: np.ndarray,
                                f_c: float = 300e9) -> Dict[str, np.ndarray]:
        """Identify operational regions in SNR-Gamma_eff space."""
        SNR, GAMMA = np.meshgrid(snr_dB_array, gamma_eff_array)
        
        # Initialize region map
        regions = np.zeros_like(SNR)
        
        for i in range(SNR.shape[0]):
            for j in range(SNR.shape[1]):
                snr_linear = 10**(SNR[i,j]/10)
                gamma_eff = GAMMA[i,j]
                
                # Calculate hardware limit SNR
                snr_hw_limit_dB = DerivedParameters.find_snr_for_hardware_limit(gamma_eff, 0.95)
                
                if SNR[i,j] < snr_hw_limit_dB - 10:
                    regions[i,j] = 1  # Power-limited
                elif SNR[i,j] > snr_hw_limit_dB:
                    regions[i,j] = 3  # Hardware-limited
                else:
                    regions[i,j] = 2  # Transition
                    
        return {'SNR': SNR, 'GAMMA': GAMMA, 'regions': regions}
    
    def plot_combined_hardware_snr_analysis(self):
        """Combined plot: Hardware sweep and SNR sweep in one figure."""
        print("\n=== Generating Combined Hardware-SNR Analysis ===")
        
        fig = plt.figure(figsize=(18, 10))
        
        # Left panel: Hardware parameter scan
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 4)
        ax4 = plt.subplot(2, 3, 5)
        
        # Right panel: SNR scan
        ax5 = plt.subplot(2, 3, (3, 6))
        
        # Parameters
        f_c = 300e9
        antenna_diameter = 1.0
        tx_power_dBm = 30
        
        # === Hardware parameter scan (left panels) ===
        gamma_eff_range = np.logspace(-3, -1, 30)
        fixed_snr_dB = [10, 20, 30, 40]
        
        axes_hw = [ax1, ax2, ax3, ax4]
        for idx, (snr_dB, ax) in enumerate(zip(fixed_snr_dB, axes_hw)):
            snr_linear = 10**(snr_dB/10)
            
            ranging_rmse_mm = []
            velocity_rmse_ms = []
            capacity_bits = []
            
            for gamma_eff in gamma_eff_range:
                # Create temporary profile
                temp_profile = HARDWARE_PROFILES["Custom"]
                original_gamma = temp_profile.Gamma_eff
                temp_profile.Gamma_eff = gamma_eff
                
                # Calculate channel parameters
                g = self.calculate_channel_gain(scenario.R_default, f_c, antenna_diameter)
                B = self.calculate_bussgang_gain()
                
                # Calculate effective noise
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance(
                    snr_linear, g, "Custom", tx_power_dBm=tx_power_dBm
                )
                
                # Calculate BCRLBs
                bcrlbs = self.calculate_observable_bcrlb(
                    f_c, sigma_eff_sq, simulation.n_pilots,
                    g, B, temp_profile.phase_noise_variance
                )
                
                ranging_rmse_mm.append(np.sqrt(bcrlbs['range']) * 1000)
                velocity_rmse_ms.append(np.sqrt(bcrlbs['range_rate']))
                
                # Calculate capacity
                phase_factor = np.exp(-temp_profile.phase_noise_variance)
                sinr_eff = snr_linear / (1 + snr_linear * gamma_eff)
                capacity = np.log2(1 + sinr_eff * phase_factor)
                capacity_bits.append(capacity)
                
                temp_profile.Gamma_eff = original_gamma
            
            # Plot ranging performance
            color = colors[idx % len(colors)]
            ax.loglog(gamma_eff_range, ranging_rmse_mm, 
                     color=color, linewidth=2.5, label=f'Range RMSE @ {snr_dB}dB')
            
            # Mark current hardware profiles
            for name, profile in HARDWARE_PROFILES.items():
                if name != "Custom":
                    ax.axvline(x=profile.Gamma_eff, color='gray', 
                              linestyle=':', alpha=0.5)
            
            # Add performance threshold
            ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
            
            # Formatting
            ax.set_xlabel('Γ_eff', fontsize=10)
            ax.set_ylabel('Range RMSE [mm]', fontsize=10)
            ax.set_title(f'SNR = {snr_dB} dB', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # === SNR scan (right panel) ===
        snr_dB_extended = np.linspace(-10, 50, 61)
        
        for idx, (profile_name, profile) in enumerate(HARDWARE_PROFILES.items()):
            if profile_name == "Custom":
                continue
                
            ranging_rmse_mm = []
            capacity_bits = []
            
            for snr_dB in snr_dB_extended:
                snr_linear = 10**(snr_dB/10)
                
                g = self.calculate_channel_gain(scenario.R_default, f_c, antenna_diameter)
                B = self.calculate_bussgang_gain()
                
                sigma_eff_sq, N_0 = self.calculate_effective_noise_variance(
                    snr_linear, g, profile_name, tx_power_dBm=tx_power_dBm
                )
                
                bcrlbs = self.calculate_observable_bcrlb(
                    f_c, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance
                )
                
                ranging_rmse_mm.append(np.sqrt(bcrlbs['range']) * 1000)
                
                # Calculate capacity
                phase_factor = np.exp(-profile.phase_noise_variance)
                sinr_eff = snr_linear / (1 + snr_linear * profile.Gamma_eff)
                capacity = np.log2(1 + sinr_eff * phase_factor)
                capacity_bits.append(capacity)
            
            color = colors[idx % len(colors)]
            ax5.semilogy(snr_dB_extended, ranging_rmse_mm,
                        color=color, linewidth=2.5,
                        label=f'{profile_name.replace("_", " ")} (Γ={profile.Gamma_eff})')
            
            # Find and mark transition SNR
            hw_limit_snr = DerivedParameters.find_snr_for_hardware_limit(
                profile.Gamma_eff, 0.95
            )
            ax5.axvline(x=hw_limit_snr, color=color, linestyle=':', alpha=0.3)
        
        # Add operational regions
        ax5.axvspan(-10, 10, alpha=0.1, color='blue', label='Power-limited')
        ax5.axvspan(30, 50, alpha=0.1, color='red', label='Hardware-limited')
        ax5.axvspan(10, 30, alpha=0.1, color='yellow', label='Transition')
        
        # Formatting
        ax5.set_xlabel('SNR [dB]', fontsize=12)
        ax5.set_ylabel('Ranging RMSE [mm]', fontsize=12)
        ax5.set_title('Performance vs SNR for Different Hardware', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper right', fontsize=9)
        ax5.set_ylim(1e-2, 1e3)
        
        # Add performance threshold
        ax5.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
        ax5.text(45, 1.2, 'Sub-mm', fontsize=10, color='green')
        
        plt.suptitle('THz ISL ISAC: Hardware Parameter Impact Analysis\n' +
                    f'(f_c = {f_c/1e9:.0f} GHz, {antenna_diameter}m Antenna, {tx_power_dBm} dBm)',
                    fontsize=16)
        plt.tight_layout()
        plt.savefig('combined_hardware_snr_analysis.pdf', format='pdf', dpi=300)
        plt.savefig('combined_hardware_snr_analysis.png', format='png', dpi=300)
        plt.show()
    
    def plot_combined_operational_regions(self):
        """Combined plot: 2D operational regions and frequency scaling."""
        print("\n=== Generating Combined Operational Regions Analysis ===")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # === Left: 2D Operational Regions ===
        snr_dB_range = np.linspace(-10, 50, 30)
        gamma_eff_range = np.logspace(-3, -1, 30)
        
        regions_data = self.find_operational_regions(snr_dB_range, gamma_eff_range)
        
        # Create custom colormap
        cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'yellow', 'red'])
        bounds = [0, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot regions
        im = ax1.contourf(regions_data['SNR'], regions_data['GAMMA'], 
                         regions_data['regions'],
                         levels=bounds, cmap=cmap, norm=norm)
        
        # Add hardware profile lines
        for name, profile in HARDWARE_PROFILES.items():
            if name != "Custom":
                ax1.axhline(y=profile.Gamma_eff, color='black', 
                          linewidth=2, label=name.replace('_', ' '))
        
        # Add iso-capacity lines
        snr_vals = np.linspace(0, 50, 100)
        for target_cap in [1, 2, 3, 4]:
            gamma_vals = 1 / (10**(snr_vals/10) * (2**target_cap - 1))
            mask = (gamma_vals >= 1e-3) & (gamma_vals <= 1e-1)
            ax1.plot(snr_vals[mask], gamma_vals[mask], 'k--', alpha=0.3)
            if np.any(mask):
                idx = np.where(mask)[0][len(np.where(mask)[0])//2]
                ax1.text(snr_vals[idx], gamma_vals[idx]*1.5, f'{target_cap} bit/s', 
                       rotation=-45, fontsize=8, alpha=0.7)
        
        ax1.set_xlabel('SNR [dB]', fontsize=12)
        ax1.set_ylabel('Hardware Quality Factor Γ_eff', fontsize=12)
        ax1.set_yscale('log')
        ax1.set_title('Operational Regions Map', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left', fontsize=8)
        
        # === Right: Frequency Scaling ===
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
        
        # Plot on twin axes
        ax2_twin = ax2.twinx()
        
        # Plot ranging performance
        valid_mask = ~np.isnan(ranging_rmse_mm)
        line1 = ax2.loglog(frequencies_GHz[valid_mask], 
                          np.array(ranging_rmse_mm)[valid_mask],
                          'b-', linewidth=3, marker='o', markersize=8, 
                          label='Ranging RMSE')
        
        # Add theoretical f^2 scaling
        f_ref = 300
        rmse_ref = np.array(ranging_rmse_mm)[frequencies_GHz == f_ref][0]
        theoretical = rmse_ref * (f_ref / frequencies_GHz)**2
        line2 = ax2.loglog(frequencies_GHz, theoretical, 'b--', linewidth=2, 
                          alpha=0.5, label='f² scaling')
        
        # Plot link margin
        line3 = ax2_twin.plot(frequencies_GHz, link_margin_dB, 'r-', linewidth=2.5,
                             marker='s', markersize=6, label='Link Margin')
        
        # Add zero margin line
        ax2_twin.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        
        # Mark invalid regions
        invalid_mask = np.array(link_margin_dB) < 0
        if np.any(invalid_mask):
            for f, valid in zip(frequencies_GHz, ~invalid_mask):
                if not valid:
                    ax2.axvspan(f*0.9, f*1.1, alpha=0.2, color='red')
        
        ax2.set_xlabel('Frequency [GHz]', fontsize=12)
        ax2.set_ylabel('Ranging RMSE [mm]', fontsize=12, color='blue')
        ax2_twin.set_ylabel('Link Margin [dB]', fontsize=12, color='red')
        ax2.set_title('Frequency Scaling Analysis', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(90, 1100)
        
        # Color the axes
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        plt.suptitle('THz ISL ISAC: Operational Regions and Frequency Scaling',
                    fontsize=16)
        plt.tight_layout()
        plt.savefig('combined_operational_analysis.pdf', format='pdf', dpi=300)
        plt.savefig('combined_operational_analysis.png', format='png', dpi=300)
        plt.show()
    
    def generate_summary_table(self):
        """Generate summary table of key results."""
        print("\n=== Performance Summary Table ===")
        print("-" * 100)
        print(f"{'Profile':<20} {'Gamma_eff':<12} {'Range@20dB':<15} {'Velocity@20dB':<15} {'Cap Ceiling':<12} {'HW Limit SNR':<12}")
        print(f"{'':20} {'':12} {'[mm]':<15} {'[m/s]':<15} {'[bits/sym]':<12} {'[dB]':<12}")
        print("-" * 100)
        
        # Fixed parameters
        snr_dB = 20
        snr_linear = 10**(snr_dB/10)
        f_c = 300e9
        antenna_diameter = 1.0
        
        for name, profile in HARDWARE_PROFILES.items():
            if name == "Custom":
                continue
                
            # Calculate performance
            g = self.calculate_channel_gain(scenario.R_default, f_c, antenna_diameter)
            B = self.calculate_bussgang_gain()
            
            sigma_eff_sq, _ = self.calculate_effective_noise_variance(
                snr_linear, g, name, tx_power_dBm=30
            )
            
            bcrlbs = self.calculate_observable_bcrlb(
                f_c, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            
            range_rmse_mm = np.sqrt(bcrlbs['range']) * 1000
            velocity_rmse_ms = np.sqrt(bcrlbs['range_rate'])
            
            ceiling = DerivedParameters.capacity_ceiling(
                profile.Gamma_eff, profile.phase_noise_variance
            )
            
            hw_limit_snr = DerivedParameters.find_snr_for_hardware_limit(
                profile.Gamma_eff, 0.95
            )
            
            print(f"{name:<20} {profile.Gamma_eff:<12.4f} {range_rmse_mm:<15.3f} "
                  f"{velocity_rmse_ms:<15.4f} {ceiling:<12.2f} {hw_limit_snr:<12.1f}")
        
        print("-" * 100)

def main():
    """Main function to run all enhanced CRLB analyses."""
    print("=== Enhanced THz ISL ISAC CRLB Analysis (Combined Visualizations) ===")
    print("\nThis analysis addresses:")
    print("1. Single ISL observability limitations (only 2 observable parameters)")
    print("2. Comprehensive hardware parameter scanning")
    print("3. Extended SNR range analysis")
    print("4. Operational region identification")
    print("5. Frequency scaling effects\n")
    
    analyzer = EnhancedCRLBAnalyzer()
    
    # Generate combined analyses
    analyzer.plot_combined_hardware_snr_analysis()
    analyzer.plot_combined_operational_regions()
    analyzer.generate_summary_table()
    
    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("1. combined_hardware_snr_analysis.png - Hardware parameter scan + SNR sweep")
    print("2. combined_operational_analysis.png - 2D regions + frequency scaling")
    
    print("\nKey Insights:")
    print("- Single ISL can only estimate range and range-rate (2 DOF)")
    print("- Hardware limitations dominate above SNR ≈ 10/Gamma_eff dB")
    print("- Sub-mm ranging requires Gamma_eff < 0.01 at reasonable SNR")
    print("- Frequency scaling provides f² improvement but limited by link budget")

if __name__ == "__main__":
    main()