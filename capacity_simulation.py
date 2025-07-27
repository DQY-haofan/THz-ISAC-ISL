#!/usr/bin/env python3
"""
capacity_simulation.py - IEEE Publication Style with Individual Plots
Updated with pointing error model and improved 2D visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
from typing import Tuple, Dict, List
from scipy.linalg import inv
from tqdm import tqdm
import os

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

class EnhancedISACSystem:
    """Enhanced THz ISL ISAC system with IEEE publication style and pointing error."""
    
    def __init__(self, hardware_profile: str, f_c: float = 300e9, 
                 distance: float = 2000e3, n_pilots: int = 64,
                 antenna_diameter: float = 1.0,
                 tx_power_dBm: float = 30):
        """Initialize with enhanced parameters."""
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        self.antenna_diameter = antenna_diameter
        self.tx_power_dBm = tx_power_dBm
        
        # Calculate system parameters
        self.lambda_c = PhysicalConstants.c / f_c
        self._calculate_link_budget()
        
        # PA parameters
        self.bussgang_gain = self._calculate_bussgang_gain()
        
        # Constellation
        self.constellation = self._create_constellation()
    
    def _calculate_link_budget(self):
        """Calculate link budget with given parameters."""
        self.P_tx_watts = 10**(self.tx_power_dBm/10) / 1000
        
        # Antenna gains
        G_single = scenario.antenna_gain(self.antenna_diameter, self.f_c)
        self.G_tx_dB = 10 * np.log10(G_single)
        self.G_rx_dB = self.G_tx_dB
        
        # Path loss
        self.path_loss_dB = DerivedParameters.path_loss_dB(self.distance, self.f_c)
        
        # Link budget
        link_budget = DerivedParameters.link_budget_dB(
            self.tx_power_dBm, self.G_tx_dB, self.G_rx_dB,
            self.distance, self.f_c
        )
        self.P_rx_dBm = link_budget['rx_power_dBm']
        self.P_rx_watts = 10**(self.P_rx_dBm/10) / 1000
        
        # Noise parameters
        self.noise_figure_dB = 8
        self.bandwidth_Hz = self.profile.signal_bandwidth_Hz
        self.noise_power_watts = DerivedParameters.thermal_noise_power(
            self.bandwidth_Hz, noise_figure_dB=self.noise_figure_dB
        )
        self.N_0 = self.noise_power_watts
        self.noise_power_dBm = 10 * np.log10(self.noise_power_watts * 1000)
        
        # Channel gain
        path_gain_linear = 10**(-self.path_loss_dB/20)
        antenna_gain_linear = 10**((self.G_tx_dB + self.G_rx_dB)/20)
        self.channel_gain = path_gain_linear * np.sqrt(antenna_gain_linear)
        
        # Link margin
        self.link_margin_dB = self.P_rx_dBm - self.noise_power_dBm
    
    def _calculate_bussgang_gain(self) -> float:
        """Calculate Bussgang gain for PA nonlinearity."""
        kappa = 10**(-7.0/10)  # 7 dB IBO
        B = 1 - 1.5 * kappa + 1.875 * kappa**2
        return B
    
    def _create_constellation(self, modulation: str = 'QPSK') -> np.ndarray:
        """Create normalized constellation."""
        if modulation == 'QPSK':
            angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            constellation = np.exp(1j * angles)
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation
    
    def calculate_sinr_mc(self, symbol: complex, avg_power: float, P_tx_scale: float,
                         n_mc: int = 100) -> float:
        """Calculate SINR for given symbol with Monte Carlo pointing error averaging."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        symbol_power = np.abs(symbol)**2
        
        # Monte Carlo averaging over pointing error
        pointing_losses = scenario.sample_pointing_loss(
            self.f_c, self.antenna_diameter, n_samples=n_mc
        )
        
        P_rx_signal_base = P_tx * symbol_power * (self.channel_gain**2) * (self.bussgang_gain**2)
        P_rx_signal_avg = P_rx_signal_base * np.mean(pointing_losses)
        
        N_thermal = self.N_0
        N_hw = P_rx_signal_avg * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        
        N_total = N_thermal + N_hw * phase_penalty
        sinr = P_rx_signal_avg / N_total
        return sinr
    
    def calculate_mutual_information(self, p_x: np.ndarray, P_tx_scale: float = 1.0,
                                   n_mc: int = 100) -> np.ndarray:
        """Calculate mutual information for each symbol with MC averaging."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr_mc(symbol, avg_power, P_tx_scale, n_mc)
            I_x[i] = np.log2(1 + sinr)
            
        return I_x
    
    def calculate_capacity_vs_snr(self, snr_dB_array: np.ndarray, n_mc: int = 100) -> Dict[str, np.ndarray]:
        """Calculate capacity vs SNR showing hardware ceiling with MC averaging."""
        capacities = []
        
        for snr_dB in tqdm(snr_dB_array, desc="    SNR sweep", leave=False):
            snr_linear = 10**(snr_dB/10)
            
            # Monte Carlo averaging
            pointing_losses = scenario.sample_pointing_loss(
                self.f_c, self.antenna_diameter, n_samples=n_mc
            )
            avg_pointing_loss = np.mean(pointing_losses)
            
            P_signal = self.P_tx_watts * (self.channel_gain**2) * (self.bussgang_gain**2) * avg_pointing_loss
            N_0_target = P_signal / snr_linear
            N_hw = P_signal * self.profile.Gamma_eff * np.exp(self.profile.phase_noise_variance)
            N_total = N_0_target + N_hw
            
            sinr_eff = P_signal / N_total
            capacity = np.log2(1 + sinr_eff)
            capacities.append(capacity)
        
        ceiling = DerivedParameters.capacity_ceiling(
            self.profile.Gamma_eff, self.profile.phase_noise_variance
        )
        
        return {
            'snr_dB': snr_dB_array,
            'capacity': np.array(capacities),
            'ceiling': ceiling
        }
    
    def calculate_bfim_observable_mc(self, avg_power: float, P_tx_scale: float,
                                    n_mc: int = 100) -> np.ndarray:
        """Calculate B-FIM for observable parameters with MC averaging."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        
        # Monte Carlo averaging
        pointing_losses = scenario.sample_pointing_loss(
            self.f_c, self.antenna_diameter, n_samples=n_mc
        )
        avg_pointing_loss = np.mean(pointing_losses)
        
        P_rx = P_tx * (self.channel_gain**2) * (self.bussgang_gain**2) * avg_pointing_loss
        
        N_thermal = self.N_0
        N_hw = P_rx * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        N_total = N_thermal + N_hw * phase_penalty
        
        SNR_eff = P_rx / N_total
        
        # FIM for range
        phase_sensitivity = (2 * np.pi * self.f_c / PhysicalConstants.c)**2
        J_range = 2 * self.n_pilots * SNR_eff * phase_sensitivity
        
        # FIM for radial velocity
        T_CPI = 1e-3
        doppler_sensitivity = (2 * np.pi * self.f_c * T_CPI / PhysicalConstants.c)**2
        J_velocity = 2 * self.n_pilots * SNR_eff * doppler_sensitivity
        
        # 2x2 FIM for observable parameters only
        J_B = np.diag([J_range, J_velocity])
        J_B += 1e-20 * np.eye(2)
        
        return J_B
    
    def calculate_distortion(self, p_x: np.ndarray, P_tx_scale: float = 1.0,
                           n_mc: int = 100) -> float:
        """Calculate sensing distortion with MC averaging."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        if avg_power < 1e-10:
            return 1e10
        
        J_B = self.calculate_bfim_observable_mc(avg_power, P_tx_scale, n_mc)
        
        try:
            J_B_inv = inv(J_B)
            distortion = J_B_inv[0,0]  # Range variance only
            
            if distortion < 0 or distortion > 1e15:
                return 1e10
                
        except np.linalg.LinAlgError:
            return 1e10
            
        return distortion

# =========================================================================
# INDIVIDUAL PLOT FUNCTIONS
# =========================================================================

def plot_cd_frontier(save_name='fig_cd_frontier'):
    """Plot C-D frontier for all hardware profiles - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        print(f"  Processing {profile_name}...")
        system = EnhancedISACSystem(profile_name)
        
        # Generate C-D points
        n_points = 12
        distortions = []
        capacities = []
        
        # Find distortion range
        p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
        D_max = system.calculate_distortion(p_uniform, n_mc=50)
        D_min = D_max / 1000
        
        D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_points)
        
        for D_target in tqdm(D_targets, desc=f"    D targets", leave=False):
            capacity, p_opt = simple_cd_optimization(system, D_target, n_mc=50)
            actual_D = system.calculate_distortion(p_opt, n_mc=50)
            
            if 0 < actual_D < 1e10 and capacity > 0:
                distortions.append(actual_D)
                capacities.append(capacity)
        
        if len(distortions) > 0:
            ranging_rmse_mm = np.sqrt(distortions) * 1000
            
            profile = HARDWARE_PROFILES[profile_name]
            
            ax.plot(ranging_rmse_mm, capacities,
                   color=colors[idx], 
                   linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                   marker=markers[idx], 
                   markersize=IEEEStyle.LINE_PROPS['markersize'],
                   markerfacecolor='white', 
                   markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                   label=f'{profile_name.replace("_", " ")}')
    
    # Add feasibility regions
    ax.axhspan(2.0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.axvspan(ax.get_xlim()[0], 1.0, alpha=0.1, color='blue')
    
    # Add performance thresholds
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(10, 2.1, 'Good communication', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='green')
    
    ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.8, 3.5, 'Sub-mm\nsensing', ha='right',
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='blue')
    
    # Labels
    ax.set_xlabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Capacity-Distortion Trade-off (All Hardware Profiles)', 
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.set_xscale('log')
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_xlim(left=0.01)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def plot_capacity_vs_snr(save_name='fig_capacity_vs_snr'):
    """Plot capacity vs SNR for all hardware profiles - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profiles = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    snr_dB_array = np.linspace(-10, 50, 60)
    
    for idx, profile_name in enumerate(profiles):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        print(f"  Processing {profile_name}...")
        system = EnhancedISACSystem(profile_name)
        results = system.calculate_capacity_vs_snr(snr_dB_array, n_mc=100)
        
        # Plot capacity
        ax.plot(snr_dB_array, results['capacity'], 
               color=colors[idx], 
               linewidth=IEEEStyle.LINE_PROPS['linewidth'],
               linestyle=linestyles[idx],
               label=profile_name.replace('_', ' '))
        
        # Add ceiling
        ax.axhline(y=results['ceiling'], color=colors[idx], 
                  linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Mark transition point
        hw_limit_snr = DerivedParameters.find_snr_for_hardware_limit(
            system.profile.Gamma_eff, 0.95
        )
        ax.axvline(x=hw_limit_snr, color=colors[idx], 
                  linestyle=':', alpha=0.3, linewidth=1.2)
    
    # Add regions
    ax.axvspan(-10, 10, alpha=0.1, color='blue')
    ax.axvspan(30, 50, alpha=0.1, color='red')
    ax.text(0, 0.5, 'Power\nLimited', ha='center', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'])
    ax.text(40, 0.5, 'Hardware\nLimited', ha='center', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'])
    
    # Labels
    ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Capacity vs. SNR with Hardware Limitations',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='lower right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(0, 8)
    ax.set_xlim(-10, 50)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def plot_2d_performance_analysis(save_name='fig_2d_performance_analysis'):
    """Plot 2D performance analysis instead of 3D landscape - NEW."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Common parameters
    hardware_profile = "High_Performance"
    
    # 1. Capacity vs Frequency (different distances)
    frequencies_GHz = np.linspace(100, 600, 20)
    distances_km = [500, 1000, 2000, 5000]
    
    for idx, d_km in enumerate(distances_km):
        capacities = []
        print(f"  Processing distance {d_km} km...")
        
        for f_GHz in frequencies_GHz:
            system = EnhancedISACSystem(
                hardware_profile, f_c=f_GHz*1e9, distance=d_km*1e3
            )
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacities.append(np.mean(I_x))
        
        ax1.plot(frequencies_GHz, capacities, 
                color=colors[idx], linewidth=2,
                marker=markers[idx], markersize=5, markevery=5,
                label=f'{d_km} km')
    
    ax1.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_title('(a) Capacity vs. Frequency', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax1.grid(True, **IEEEStyle.GRID_PROPS)
    ax1.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax1.set_ylim(bottom=0)
    
    # 2. Capacity vs Distance (different frequencies)
    distances_km = np.linspace(500, 5000, 20)
    frequencies_GHz = [100, 200, 300, 600]
    
    for idx, f_GHz in enumerate(frequencies_GHz):
        capacities = []
        
        for d_km in distances_km:
            system = EnhancedISACSystem(
                hardware_profile, f_c=f_GHz*1e9, distance=d_km*1e3
            )
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacities.append(np.mean(I_x))
        
        ax2.plot(distances_km, capacities, 
                color=colors[idx], linewidth=2,
                marker=markers[idx], markersize=5, markevery=5,
                label=f'{f_GHz} GHz')
    
    ax2.set_xlabel('Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_title('(b) Capacity vs. Distance', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax2.grid(True, **IEEEStyle.GRID_PROPS)
    ax2.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax2.set_ylim(bottom=0)
    
    # 3. RMSE vs Frequency (different SNRs)
    frequencies_GHz = np.linspace(100, 600, 20)
    snr_dB_values = [10, 20, 30, 40]
    
    for idx, snr_dB in enumerate(snr_dB_values):
        rmse_values = []
        
        for f_GHz in frequencies_GHz:
            system = EnhancedISACSystem(hardware_profile, f_c=f_GHz*1e9)
            # Simplified RMSE calculation
            snr_linear = 10**(snr_dB/10)
            phase_sensitivity = (2 * np.pi * f_GHz*1e9 / PhysicalConstants.c)**2
            crlb = 1 / (2 * system.n_pilots * snr_linear * phase_sensitivity)
            rmse_mm = np.sqrt(crlb) * 1000
            rmse_values.append(rmse_mm)
        
        ax3.semilogy(frequencies_GHz, rmse_values, 
                    color=colors[idx], linewidth=2,
                    marker=markers[idx], markersize=5, markevery=5,
                    label=f'{snr_dB} dB')
    
    ax3.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax3.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax3.set_title('(c) RMSE vs. Frequency', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax3.grid(True, **IEEEStyle.GRID_PROPS)
    ax3.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax3.set_ylim(1e-3, 1e2)
    
    # 4. RMSE vs SNR (different frequencies)
    snr_dB_array = np.linspace(0, 40, 20)
    frequencies_GHz = [100, 200, 300, 600]
    
    for idx, f_GHz in enumerate(frequencies_GHz):
        rmse_values = []
        
        for snr_dB in snr_dB_array:
            system = EnhancedISACSystem(hardware_profile, f_c=f_GHz*1e9)
            snr_linear = 10**(snr_dB/10)
            phase_sensitivity = (2 * np.pi * f_GHz*1e9 / PhysicalConstants.c)**2
            crlb = 1 / (2 * system.n_pilots * snr_linear * phase_sensitivity)
            rmse_mm = np.sqrt(crlb) * 1000
            rmse_values.append(rmse_mm)
        
        ax4.semilogy(snr_dB_array, rmse_values, 
                    color=colors[idx], linewidth=2,
                    marker=markers[idx], markersize=5, markevery=5,
                    label=f'{f_GHz} GHz')
    
    ax4.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax4.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax4.set_title('(d) RMSE vs. SNR', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax4.grid(True, **IEEEStyle.GRID_PROPS)
    ax4.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax4.set_ylim(1e-3, 1e2)
    
    plt.suptitle('THz ISL ISAC Performance Analysis', fontsize=IEEEStyle.FONT_SIZES['title']+2)
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def plot_performance_summary(save_name='fig_performance_summary'):
    """Plot performance summary comparison without bar value labels."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profiles = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    # Calculate metrics at SNR = 20 dB
    ranging_rmse = []
    capacity = []
    hw_limit_snr = []
    
    for profile_name in profiles:
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        profile = HARDWARE_PROFILES[profile_name]
        
        # Simplified calculations
        snr_linear = 100  # 20 dB
        f_c = 300e9
        
        # Include pointing error effect
        pointing_loss = scenario.calculate_pointing_loss_factor(f_c, 1.0)
        
        rmse = 1000 * np.sqrt(1 / (snr_linear * (f_c/3e8)**2 * profile.Gamma_eff**(-1) * pointing_loss))
        ranging_rmse.append(rmse)
        
        cap = np.log2(1 + snr_linear / (1 + snr_linear * profile.Gamma_eff))
        capacity.append(cap)
        
        hw_snr = DerivedParameters.find_snr_for_hardware_limit(profile.Gamma_eff, 0.95)
        hw_limit_snr.append(hw_snr)
    
    # Create grouped bar chart
    x = np.arange(len(profiles))
    width = 0.25
    
    # Normalize metrics for display
    ranging_norm = np.array(ranging_rmse) / max(ranging_rmse)
    capacity_norm = np.array(capacity) / max(capacity)
    hw_snr_norm = np.array(hw_limit_snr) / max(hw_limit_snr)
    
    bars1 = ax.bar(x - width, ranging_norm, width, 
                   label='Ranging RMSE', color=colors[0], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, capacity_norm, width, 
                   label='Capacity', color=colors[1], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, hw_snr_norm, width, 
                   label='HW Limit SNR', color=colors[2], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Labels
    ax.set_xlabel('Hardware Profile', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Normalized Performance', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Performance Summary at SNR = 20 dB',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in profiles], 
                      fontsize=IEEEStyle.FONT_SIZES['tick'])
    ax.legend(loc='upper left', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(0, 1.2)
    ax.grid(True, axis='y', **IEEEStyle.GRID_PROPS)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def simple_cd_optimization(system: EnhancedISACSystem, D_target: float,
                          max_iterations: int = 20, n_mc: int = 50) -> Tuple[float, np.ndarray]:
    """Simplified C-D optimization with MC averaging."""
    n_symbols = len(system.constellation)
    
    # Initialize with uniform distribution
    p_x = np.ones(n_symbols) / n_symbols
    
    # Check if already meets constraint
    D_current = system.calculate_distortion(p_x, n_mc=n_mc)
    if D_current <= D_target:
        I_x = system.calculate_mutual_information(p_x, n_mc=n_mc)
        return np.sum(p_x * I_x), p_x
    
    # Simple gradient-based optimization
    step_size = 0.1
    for _ in range(max_iterations):
        # Calculate gradient
        I_x = system.calculate_mutual_information(p_x, n_mc=n_mc)
        
        # Update towards higher power symbols if distortion too high
        if D_current > D_target:
            symbol_powers = np.abs(system.constellation)**2
            gradient = symbol_powers - np.mean(symbol_powers)
        else:
            gradient = I_x - np.mean(I_x)
        
        # Update distribution
        p_x = p_x * np.exp(step_size * gradient)
        p_x /= np.sum(p_x)
        
        # Check new distortion
        D_current = system.calculate_distortion(p_x, n_mc=n_mc)
    
    capacity = np.sum(p_x * I_x)
    return capacity, p_x

def main():
    """Main function to generate all capacity analysis plots."""
    print("=== THz ISL ISAC Capacity Analysis (IEEE Style) ===")
    print("With pointing error Monte Carlo averaging")
    print(f"Current font sizes: {IEEEStyle.FONT_SIZES}")
    
    # Note about observability
    print("\nNOTE: Analysis based on single ISL (2 observable parameters)")
    
    # Generate all individual plots
    plot_cd_frontier()
    plot_capacity_vs_snr()
    plot_2d_performance_analysis()  # Replaces 3D plots
    plot_performance_summary()
    
    print("\n=== Capacity Analysis Complete ===")
    print("Generated files in results/:")
    print("- fig_cd_frontier.pdf/png")
    print("- fig_capacity_vs_snr.pdf/png")
    print("- fig_2d_performance_analysis.pdf/png")
    print("- fig_performance_summary.pdf/png")

if __name__ == "__main__":
    main()