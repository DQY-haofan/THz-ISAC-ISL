#!/usr/bin/env python3
"""
crlb_simulation.py - IEEE Publication Style with Individual Plots
Updated with data saving, 3D plots, and 1THz support
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, Dict, List
import itertools
import os
from tqdm import tqdm
from scipy.special import erf, erfc
import numpy as np

# Import configuration
from simulation_config import (
    PhysicalConstants, 
    scenario, 
    simulation,
    HARDWARE_PROFILES,
    DerivedParameters,
    ObservableParameters,
    IEEEStyle,
    data_saver
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
                              antenna_diameter: float = None) -> float:
        """Calculate channel gain magnitude |g|."""
        if antenna_diameter is None:
            antenna_diameter = scenario.default_antenna_diameter
        lambda_c = PhysicalConstants.wavelength(frequency_Hz)
        G_single = scenario.antenna_gain(antenna_diameter, frequency_Hz)
        beta_ch = (lambda_c / (4 * np.pi * distance_m)) * np.sqrt(G_single * G_single)
        return beta_ch
    
    def calculate_bussgang_gain(self, input_backoff_dB: float = 7.0) -> float:
        """Calculate Bussgang gain for PA nonlinearity."""
        kappa = 10 ** (-input_backoff_dB / 10)
        B = 1 - 1.5 * kappa + 1.875 * kappa**2
        # Ensure real value
        return float(np.real(B))
    
    def calculate_effective_noise_variance_mc(
        self, SNR_linear: float, channel_gain: float, hardware_profile: str,
        signal_power: float = 1.0, tx_power_dBm: float = None,
        bandwidth_Hz: float = 10e9, frequency_Hz: float = 300e9,
        antenna_diameter: float = None, n_mc: int = 100
    ) -> Tuple[float, float, float]:
        """Calculate effective noise variance targeting specific SNR.
        
        Returns:
            Tuple of (sigma_eff_sq, N_thermal, SNR_eff)
        """
        if tx_power_dBm is None:
            tx_power_dBm = scenario.default_tx_power_dBm
        if antenna_diameter is None:
            antenna_diameter = scenario.default_antenna_diameter
            
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        
        P_tx_watts = 10**(tx_power_dBm/10) / 1000
        
        # Use expected pointing loss (analytical formula)
        pointing_loss_expected = scenario.calculate_pointing_loss_factor(
            frequency_Hz, antenna_diameter
        )
        
        # Calculate received power with expected pointing loss
        P_rx = P_tx_watts * signal_power * (channel_gain ** 2) * (B ** 2) * pointing_loss_expected
        
        # CRITICAL FIX: Set thermal noise to achieve target SNR
        # This makes the CRLB actually vary with SNR
        N_awgn = P_rx / SNR_linear  # Target AWGN noise for requested SNR
        
        # Hardware-dependent noise
        N_hw = P_rx * profile.Gamma_eff * np.exp(profile.phase_noise_variance)
        
        # DSE residual (parameterized)
        N_DSE = simulation.kappa_DSE * N_awgn
        
        # Total effective noise
        N_total = N_awgn + N_hw + N_DSE
        
        # Effective SNR (for FIM calculation)
        SNR_eff = P_rx / N_total
        
        return N_total, N_awgn, SNR_eff

    
    def calculate_observable_bcrlb_mc(
        self, f_c: float, sigma_eff_sq: float, M: int,
        channel_gain: float, B: float, sigma_phi_sq: float,
        T_CPI: float = 1e-3, signal_power: float = 1.0,
        antenna_diameter: float = None, n_mc: int = 100,
        SNR_eff: float = None  # 新增参数
    ) -> Dict[str, float]:
        """Calculate BCRLB using consistent SNR_eff.
        
        Now uses SNR_eff directly instead of recalculating with MC.
        """
        if antenna_diameter is None:
            antenna_diameter = scenario.default_antenna_diameter
        
        # Phase and Doppler sensitivity terms
        phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
        doppler_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2 * T_CPI**2)
        
        # Phase noise penalty
        phase_penalty = np.exp(sigma_phi_sq)
        
        # BCRLB using SNR_eff
        bcrlb_range = phase_term * phase_penalty / (M * SNR_eff)
        bcrlb_range_rate = doppler_term * phase_penalty / (M * SNR_eff)
        
        return {
            'range': bcrlb_range,
            'range_rate': bcrlb_range_rate
        }
    
    # =========================================================================
    # INDIVIDUAL PLOT FUNCTIONS
    # =========================================================================
    
 # 文件: crlb_simulation.py
# 修改 plot_ranging_crlb_vs_snr 函数

    def plot_ranging_crlb_vs_snr(self, save_name='fig_ranging_crlb_vs_snr'):
        """Plot ranging CRLB vs SNR with auto-scaled axes."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
        
        snr_dB = np.linspace(-10, 60, 71)
        frequency_Hz = 300e9
        
        profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
        
        data_to_save = {
            'snr_dB': snr_dB.tolist(),
            'frequency_GHz': frequency_Hz/1e9,
            'hardware_profiles': profiles_to_plot
        }
        
        # Track data ranges
        all_rmse = []
        
        for i, hardware_profile in enumerate(profiles_to_plot):
            if hardware_profile not in HARDWARE_PROFILES:
                continue
            
            profile = HARDWARE_PROFILES[hardware_profile]
            
            # Calculate RMSE for each SNR
            rmse_mm = []
            for snr_point in 10**(snr_dB/10):
                sigma_eff_sq, _, SNR_eff = self.calculate_effective_noise_variance_mc(
                    snr_point, 1.0, hardware_profile, 1.0, bandwidth_Hz=10e9,
                    frequency_Hz=frequency_Hz, n_mc=50
                )
                
                phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * frequency_Hz**2)
                bcrlb = phase_term * np.exp(profile.phase_noise_variance) / (simulation.n_pilots * SNR_eff)
                rmse_mm.append(np.sqrt(bcrlb) * 1000)
            
            rmse_mm = np.array(rmse_mm)
            all_rmse.extend(rmse_mm[np.isfinite(rmse_mm)])  # Track valid values
            
            data_to_save[f'rmse_mm_{hardware_profile}'] = rmse_mm.tolist()
            
            ax.semilogy(snr_dB, rmse_mm, 
                    color=colors[i], 
                    linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                    marker=markers[i], 
                    markersize=IEEEStyle.LINE_PROPS['markersize']-1,
                    markevery=10,
                    markerfacecolor='white', 
                    markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                    label=f'{hardware_profile.replace("_", " ")}')
        
        # AUTO-SCALE y-axis based on actual data
        if all_rmse:
            valid_rmse = [r for r in all_rmse if r > 0 and np.isfinite(r)]
            if valid_rmse:
                rmse_min, rmse_max = min(valid_rmse), max(valid_rmse)
                # Add margins in log space
                log_range = np.log10(rmse_max) - np.log10(rmse_min)
                y_min = 10**(np.log10(rmse_min) - 0.1 * log_range)
                y_max = 10**(np.log10(rmse_max) + 0.3 * log_range)  # Extra space for legend
                ax.set_ylim(y_min, y_max)
        
        # Configure plot
        ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title(f'Ranging Accuracy vs SNR at {frequency_Hz/1e9:.0f} GHz',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        ax.legend(loc='best', fontsize=IEEEStyle.FONT_SIZES['legend'])
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        data_saver.save_data(save_name, data_to_save,
                        "Ranging CRLB vs SNR for all hardware profiles")
        
        print(f"Saved: results/{save_name}.pdf/png and data")

        
    def plot_ranging_vs_frequency(self, save_name='fig_ranging_vs_frequency'):
        """Plot ranging performance vs frequency (separate plot)."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
        
        # Parameters
        snr_dB = simulation.default_SNR_dB  # High SNR
        frequencies_GHz = simulation.frequency_sweep_GHz
        frequencies_Hz = frequencies_GHz * 1e9
        hardware_profile = "High_Performance"
        antenna_diameter = scenario.default_antenna_diameter
        tx_power_dBm = scenario.default_tx_power_dBm
        
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        
        # Data storage
        data_to_save = {
            'frequency_GHz': frequencies_GHz.tolist(),
            'snr_dB': snr_dB,
            'hardware_profile': hardware_profile
        }
        
        ranging_rmse_mm = []
        
        print(f"  Processing frequencies up to {frequencies_GHz[-1]} GHz...")
        for f_Hz in tqdm(frequencies_Hz, desc="    Frequency sweep", leave=False):
            snr_linear = 10 ** (snr_dB / 10)
            
            g = self.calculate_channel_gain(scenario.R_default, f_Hz, antenna_diameter)
            
            # FIX: Now expecting 3 return values
            sigma_eff_sq, N_thermal, SNR_eff = self.calculate_effective_noise_variance_mc(
                snr_linear, g, hardware_profile, tx_power_dBm=tx_power_dBm,
                frequency_Hz=f_Hz, antenna_diameter=antenna_diameter, n_mc=100
            )
            
            # Pass SNR_eff to BCRLB calculation
            bcrlbs = self.calculate_observable_bcrlb_mc(
                f_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance,
                antenna_diameter=antenna_diameter, n_mc=100,
                SNR_eff=SNR_eff  # Pass the calculated SNR_eff
            )
            
            rmse_m = np.sqrt(bcrlbs['range'])
            ranging_rmse_mm.append(rmse_m * 1000)
        
        data_to_save['ranging_rmse_mm'] = ranging_rmse_mm
        
        # Plot
        ax.loglog(frequencies_GHz, ranging_rmse_mm,
                color=colors[0], 
                linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                marker=markers[0], 
                markersize=IEEEStyle.LINE_PROPS['markersize'],
                markerfacecolor='white',
                markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                label='Actual Performance')
        
        # Add theoretical f^-2 scaling
        f_ref = 300
        rmse_ref = ranging_rmse_mm[np.where(frequencies_GHz == f_ref)[0][0]]
        theoretical = rmse_ref * (f_ref / frequencies_GHz)**2
        ax.loglog(frequencies_GHz, theoretical, 
                color=colors[0], linestyle='--', 
                linewidth=IEEEStyle.LINE_PROPS['linewidth']-0.5,
                label='$f^{-2}$ scaling')
        

        # Performance threshold
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Labels
        ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title(f'Frequency Scaling of Ranging Performance (SNR = {snr_dB} dB)',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
        
        # Set axis limits with padding
        ax.set_xlim(80, 1200)
        ax.set_ylim(1e-4, 10)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        data_saver.save_data(save_name, data_to_save,
                        "Ranging performance vs frequency scaling")
        
        print(f"Saved: results/{save_name}.pdf/png and data")


    def plot_velocity_vs_frequency(self, save_name='fig_velocity_vs_frequency'):
        """Plot velocity estimation performance vs frequency (separate plot)."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
        
        # Parameters
        snr_dB = simulation.default_SNR_dB
        frequencies_GHz = simulation.frequency_sweep_GHz
        frequencies_Hz = frequencies_GHz * 1e9
        hardware_profile = "High_Performance"
        antenna_diameter = scenario.default_antenna_diameter
        tx_power_dBm = scenario.default_tx_power_dBm
        
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        
        # Data storage
        data_to_save = {
            'frequency_GHz': frequencies_GHz.tolist(),
            'snr_dB': snr_dB,
            'hardware_profile': hardware_profile
        }
        
        velocity_rmse_ms = []
        
        for f_Hz in tqdm(frequencies_Hz, desc="    Frequency sweep", leave=False):
            snr_linear = 10 ** (snr_dB / 10)
            
            g = self.calculate_channel_gain(scenario.R_default, f_Hz, antenna_diameter)
            
            # FIX: Now expecting 3 return values
            sigma_eff_sq, N_thermal, SNR_eff = self.calculate_effective_noise_variance_mc(
                snr_linear, g, hardware_profile, tx_power_dBm=tx_power_dBm,
                frequency_Hz=f_Hz, antenna_diameter=antenna_diameter, n_mc=100
            )
            
            # Pass SNR_eff to BCRLB calculation
            bcrlbs = self.calculate_observable_bcrlb_mc(
                f_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance,
                antenna_diameter=antenna_diameter, n_mc=100,
                SNR_eff=SNR_eff  # Pass the calculated SNR_eff
            )
            
            rmse_v = np.sqrt(bcrlbs['range_rate'])
            velocity_rmse_ms.append(rmse_v)
        
        data_to_save['velocity_rmse_ms'] = velocity_rmse_ms
        
        # Plot
        ax.loglog(frequencies_GHz, velocity_rmse_ms,
                color=colors[1], 
                linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                marker=markers[1], 
                markersize=IEEEStyle.LINE_PROPS['markersize'],
                markerfacecolor='white',
                markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'])
        

        # Labels
        ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Velocity RMSE (m/s)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title(f'Velocity Estimation vs. Frequency (SNR = {snr_dB} dB)',
                    fontsize=IEEEStyle.FONT_SIZES['title'])
        ax.grid(True, **IEEEStyle.GRID_PROPS)
        
        # Set axis limits with padding
        ax.set_xlim(80, 1200)
        ax.set_ylim(1e-4, 1e0)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        data_saver.save_data(save_name, data_to_save,
                        "Velocity estimation performance vs frequency")
        
        print(f"Saved: results/{save_name}.pdf/png and data")

    def plot_pointing_error_sensitivity(self, save_name='fig_pointing_error_sensitivity'):
        """Plot sensitivity to pointing error - standalone."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
        
        # Parameters
        f_c = 300e9
        antenna_diameter = scenario.default_antenna_diameter
        tx_power_dBm = scenario.default_tx_power_dBm
        hardware_profile = "High_Performance"
        pointing_errors_urad = [0.5, 1.0, 2.0]  # µrad
        
        profile = HARDWARE_PROFILES[hardware_profile]
        B = self.calculate_bussgang_gain()
        g = self.calculate_channel_gain(scenario.R_default, f_c, antenna_diameter)
        
        # Data storage
        data_to_save = {
            'snr_dB': simulation.SNR_dB_array.tolist(),
            'pointing_errors_urad': pointing_errors_urad,
            'hardware_profile': hardware_profile
        }
        
        for idx, pe_urad in enumerate(pointing_errors_urad):
            pe_rad = pe_urad * 1e-6
            ranging_rmse_mm = []
            
            # Override scenario pointing error temporarily
            original_pe = scenario.pointing_error_rms_rad
            scenario.pointing_error_rms_rad = pe_rad
            
            print(f"  Processing σ_θ = {pe_urad} µrad...")
            for snr_dB in tqdm(simulation.SNR_dB_array, desc="    SNR sweep", leave=False):
                snr_linear = 10**(snr_dB/10)
                
                # FIX: Now expecting 3 return values
                sigma_eff_sq, N_thermal, SNR_eff = self.calculate_effective_noise_variance_mc(
                    snr_linear, g, hardware_profile, tx_power_dBm=tx_power_dBm,
                    frequency_Hz=f_c, antenna_diameter=antenna_diameter, n_mc=100
                )
                
                # Pass SNR_eff to BCRLB calculation
                bcrlbs = self.calculate_observable_bcrlb_mc(
                    f_c, sigma_eff_sq, simulation.n_pilots,
                    g, B, profile.phase_noise_variance,
                    antenna_diameter=antenna_diameter, n_mc=100,
                    SNR_eff=SNR_eff  # Pass the calculated SNR_eff
                )
                
                ranging_rmse_mm.append(np.sqrt(bcrlbs['range']) * 1000)
            
            # Restore original
            scenario.pointing_error_rms_rad = original_pe
            
            data_to_save[f'ranging_rmse_mm_{pe_urad}urad'] = ranging_rmse_mm
            
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
        ax.set_xlim(-10, 60)
        ax.set_ylim(1e-3, 1e4)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        data_saver.save_data(save_name, data_to_save,
                        "Pointing error sensitivity analysis")
        
        print(f"Saved: results/{save_name}.pdf/png and data")

    
    def plot_feasibility_map(self, save_name='fig_feasibility_map'):
        """Plot 2D feasibility map for antenna size vs transmit power."""
        print(f"\n=== Generating {save_name} ===")
        
        fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
        
        # Parameter ranges
        antenna_diameters = np.linspace(0.5, 2.0, 20)
        tx_powers_dBm = np.linspace(20, 35, 20)
        
        # Fixed parameters
        f_c = 300e9
        distance = scenario.R_default
        hardware_profile = "High_Performance"
        
        # Thresholds
        min_link_margin_dB = 3
        max_ranging_rmse_mm = 1.0
        min_capacity_bits = 2.0
        max_excellent_rmse_mm = 0.1
        min_excellent_capacity = 4.0
        
        # Create meshgrid
        D, P = np.meshgrid(antenna_diameters, tx_powers_dBm)
        
        # Initialize feasibility map
        feasibility = np.zeros_like(D)
        
        data_to_save = {
            'antenna_diameters_m': antenna_diameters.tolist(),
            'tx_powers_dBm': tx_powers_dBm.tolist(),
            'hardware_profile': hardware_profile
        }
        
        print("  Computing feasibility map...")
        for i in tqdm(range(D.shape[0]), desc="    Power levels"):
            for j in range(D.shape[1]):
                ant_diam = D[i,j]
                tx_power = P[i,j]
                
                # Calculate actual link budget
                P_tx_watts = 10**(tx_power/10) / 1000
                
                # Channel gain with specific antenna
                g = self.calculate_channel_gain(distance, f_c, ant_diam)
                B = self.calculate_bussgang_gain()
                profile = HARDWARE_PROFILES[hardware_profile]
                
                # Expected pointing loss
                pointing_loss = scenario.calculate_pointing_loss_factor(f_c, ant_diam)
                
                # Received power
                P_rx = P_tx_watts * (g**2) * (B**2) * pointing_loss
                
                # Thermal noise
                noise_figure_linear = 10**(8/10)
                N_thermal = PhysicalConstants.k * 290 * noise_figure_linear * profile.signal_bandwidth_Hz
                
                # Hardware noise
                N_hw = P_rx * profile.Gamma_eff * np.exp(profile.phase_noise_variance)
                
                # Total noise
                N_total = N_thermal + N_hw
                
                # Actual SNR_eff (not fixed!)
                SNR_eff = P_rx / N_total
                
                # Link margin check
                link_margin_dB = 10*np.log10(P_rx) - 10*np.log10(N_thermal)
                
                if link_margin_dB < min_link_margin_dB:
                    feasibility[i,j] = 0  # Link doesn't close
                    continue
                
                # Calculate capacity with actual SNR_eff
                capacity = np.log2(1 + SNR_eff)
                
                # Calculate ranging RMSE using BCRLB
                phase_term = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
                phase_penalty = np.exp(profile.phase_noise_variance)
                bcrlb_range = phase_term * phase_penalty / (simulation.n_pilots * SNR_eff)
                ranging_rmse_mm = np.sqrt(bcrlb_range) * 1000
                
                # Determine feasibility level
                comm_excellent = capacity >= min_excellent_capacity
                sense_excellent = ranging_rmse_mm <= max_excellent_rmse_mm
                comm_ok = capacity >= min_capacity_bits
                sense_ok = ranging_rmse_mm <= max_ranging_rmse_mm
                
                if comm_excellent and sense_excellent:
                    feasibility[i,j] = 4  # Excellent both
                elif comm_ok and sense_ok:
                    feasibility[i,j] = 3  # Both OK
                elif comm_ok:
                    feasibility[i,j] = 1  # Communication only
                elif sense_ok:
                    feasibility[i,j] = 2  # Sensing only
                else:
                    feasibility[i,j] = 0.5  # Link OK but neither meets specs
        
        data_to_save['feasibility_map'] = feasibility.tolist()
        
        # Create professional colormap
        colors_map = [IEEEStyle.COLORS_FEASIBILITY['infeasible'],
                     '#ffcccc',  # Link OK but poor performance
                     IEEEStyle.COLORS_FEASIBILITY['comm_only'],
                     IEEEStyle.COLORS_FEASIBILITY['sense_only'],
                     IEEEStyle.COLORS_FEASIBILITY['both'],
                     IEEEStyle.COLORS_FEASIBILITY['excellent']]
        cmap = plt.cm.colors.ListedColormap(colors_map)
        bounds = [0, 0.25, 0.75, 1.5, 2.5, 3.5, 4.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        im = ax.contourf(D, P, feasibility, levels=bounds, cmap=cmap, norm=norm)
        
        # Add contour lines
        ax.contour(D, P, feasibility, levels=[0.5, 1.5, 2.5, 3.5], colors='black', 
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
            Patch(facecolor=IEEEStyle.COLORS_FEASIBILITY['infeasible'], 
                  edgecolor='black', label='Link Fails'),
            Patch(facecolor='#ffcccc', edgecolor='black', label='Poor Performance'),
            Patch(facecolor=IEEEStyle.COLORS_FEASIBILITY['comm_only'], 
                  edgecolor='black', label='Communication Only'),
            Patch(facecolor=IEEEStyle.COLORS_FEASIBILITY['sense_only'], 
                  edgecolor='black', label='Sensing Only'),
            Patch(facecolor=IEEEStyle.COLORS_FEASIBILITY['both'], 
                  edgecolor='black', label='ISAC Feasible'),
            Patch(facecolor=IEEEStyle.COLORS_FEASIBILITY['excellent'], 
                  edgecolor='black', label='Excellent Performance')
        ]
        ax.legend(handles=legend_elements, loc='lower right', 
                 fontsize=IEEEStyle.FONT_SIZES['legend']-1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        data_saver.save_data(save_name, data_to_save,
                           "ISAC feasibility map for antenna size vs transmit power")
        
        print(f"Saved: results/{save_name}.pdf/png and data")
    
    def plot_3d_performance_landscape(self, save_name='fig_3d_performance_landscape'):
        """Generate 3D performance landscape with proper labels."""
        print(f"\n=== Generating {save_name} ===")
        
        # Generate grid data
        frequencies_GHz = np.linspace(100, 1000, 30)
        distances_km = np.linspace(500, 5000, 30)
        F, D = np.meshgrid(frequencies_GHz, distances_km)
        
        # Calculate capacity for each point
        capacity_grid = np.zeros_like(F)
        
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                freq_Hz = F[i, j] * 1e9
                dist_m = D[i, j] * 1e3
                
                # Create system and calculate capacity
                self.f_c = freq_Hz
                self.distance = dist_m
                self.lambda_c = PhysicalConstants.c / freq_Hz
                self._calculate_enhanced_link_budget()
                
                # Calculate capacity
                p_x = np.ones(len(self.constellation)) / len(self.constellation)
                I_x = self.calculate_mutual_information(p_x, P_tx_scale=1.0, n_mc=30)
                capacity_grid[i, j] = np.mean(I_x)
        
        # Create single optimized view
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(F, D, capacity_grid, 
                            cmap='viridis', 
                            edgecolor='none', 
                            alpha=0.9,
                            antialiased=True)
        
        # Add contour lines at the bottom (without red line)
        contours = ax.contour(F, D, capacity_grid, 
                            zdir='z', 
                            offset=np.min(capacity_grid) - 0.5,
                            cmap='viridis', 
                            alpha=0.5,
                            linewidths=1.0)
        
        # CRITICAL: Set all axis labels with proper spacing
        ax.set_xlabel('Frequency (GHz)', 
                    fontsize=IEEEStyle.FONT_SIZES['label'], 
                    labelpad=10)
        ax.set_ylabel('Distance (km)', 
                    fontsize=IEEEStyle.FONT_SIZES['label'], 
                    labelpad=10)
        ax.set_zlabel('Capacity (bits/symbol)', 
                    fontsize=IEEEStyle.FONT_SIZES['label'], 
                    labelpad=15)  # Extra padding for Z-label
        
        # Set title with proper spacing
        ax.set_title('THz ISL ISAC Capacity Landscape\n(High Performance Hardware)', 
                    fontsize=IEEEStyle.FONT_SIZES['title'], 
                    pad=20)
        
        # Optimize viewing angle for best visibility
        ax.view_init(elev=25, azim=45)
        
        # Adjust axis properties
        ax.set_xlim(100, 1000)
        ax.set_ylim(500, 5000)
        ax.set_zlim(np.min(capacity_grid) - 0.5, np.max(capacity_grid) + 0.5)
        
        # Format tick labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        
        # Adjust tick label size
        ax.tick_params(axis='both', which='major', 
                    labelsize=IEEEStyle.FONT_SIZES['tick'])
        ax.tick_params(axis='z', which='major', 
                    labelsize=IEEEStyle.FONT_SIZES['tick'])
        
        # Add colorbar with proper label
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Capacity (bits/symbol)', 
                    fontsize=IEEEStyle.FONT_SIZES['label'],
                    rotation=270, 
                    labelpad=20)
        cbar.ax.tick_params(labelsize=IEEEStyle.FONT_SIZES['tick'])
        
        # Set background color
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        
        # Ensure Z-label is visible by adjusting distance
        ax.dist = 11  # Default is 10, increase for better label visibility
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data
        data_to_save = {
            'frequency_GHz': frequencies_GHz.tolist(),
            'distance_km': distances_km.tolist(),
            'capacity_grid': capacity_grid.tolist(),
            'hardware_profile': 'High_Performance'
        }
        
        data_saver.save_data(save_name, data_to_save,
                        "3D capacity landscape for High Performance hardware")
        
        print(f"Saved: results/{save_name}.pdf/png and data")



def main():
    """Main function to generate all CRLB analysis plots."""
    print("=== THz ISL ISAC CRLB Analysis (IEEE Style) ===")
    print("With data saving and 3D visualizations")
    print(f"Default SNR: {simulation.default_SNR_dB} dB")
    print(f"Default antenna: {scenario.default_antenna_diameter} m")
    print(f"Default TX power: {scenario.default_tx_power_dBm} dBm")
    
    # Print observability warning once
    ObservableParameters.print_observability_warning()
    
    analyzer = EnhancedCRLBAnalyzer()
    
    # Generate all plots
    analyzer.plot_ranging_crlb_vs_snr()
    analyzer.plot_ranging_vs_frequency()
    analyzer.plot_velocity_vs_frequency()
    analyzer.plot_pointing_error_sensitivity()
    analyzer.plot_feasibility_map()
    analyzer.plot_3d_performance_landscape()
    
    print("\n=== CRLB Analysis Complete ===")
    print("Generated files in results/:")
    print("- fig_ranging_crlb_vs_snr.pdf/png + data")
    print("- fig_ranging_vs_frequency.pdf/png + data")
    print("- fig_velocity_vs_frequency.pdf/png + data")
    print("- fig_pointing_error_sensitivity.pdf/png + data")
    print("- fig_feasibility_map.pdf/png + data")
    print("- fig_3d_performance_landscape_[views].pdf/png + data")

if __name__ == "__main__":
    main()