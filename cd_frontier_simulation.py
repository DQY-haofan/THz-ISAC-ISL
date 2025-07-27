#!/usr/bin/env python3
"""
cd_frontier_simulation.py - Enhanced Version with Pointing Error and 2D Decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
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
    IEEEStyle
)

# Setup IEEE style
IEEEStyle.setup()
colors = IEEEStyle.get_colors()
markers = IEEEStyle.get_markers()
linestyles = IEEEStyle.get_linestyles()

# Global debug flag
DEBUG_VERBOSE = False

class ISACSystem:
    """Enhanced THz ISL ISAC system with pointing error model."""
    
    def __init__(self, hardware_profile: str, f_c: float = 300e9, 
                 distance: float = 2000e3, n_pilots: int = 64,
                 antenna_diameter: float = 1.0):
        """Initialize with enhanced parameters."""
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        self.antenna_diameter = antenna_diameter
        
        # Calculate system parameters
        self.lambda_c = PhysicalConstants.c / f_c
        
        # Enhanced link budget
        self._calculate_enhanced_link_budget()
        
        # PA parameters
        self.bussgang_gain = self._calculate_bussgang_gain()
        
        # Constellation
        self.constellation = self._create_constellation()
        
        # Debug counter
        self._debug_count = 0
    
    def _calculate_enhanced_link_budget(self):
        """Enhanced link budget with larger antennas and higher power."""
        # Increased transmit power for satellite
        self.P_tx_dBm = 30  # 30 dBm = 1W
        self.P_tx_watts = 10**(self.P_tx_dBm/10) / 1000
        
        # Enhanced antenna gains with larger diameter
        antenna_efficiency = 0.65
        G_single = antenna_efficiency * (np.pi * self.antenna_diameter / self.lambda_c)**2
        self.G_tx_dB = 10 * np.log10(G_single)
        self.G_rx_dB = self.G_tx_dB
        
        # Path loss
        self.path_loss_dB = 20 * np.log10(4 * np.pi * self.distance / self.lambda_c)
        
        # Total link budget
        self.P_rx_dBm = self.P_tx_dBm + self.G_tx_dB + self.G_rx_dB - self.path_loss_dB
        self.P_rx_watts = 10**(self.P_rx_dBm/10) / 1000
        
        # Noise parameters
        self.noise_figure_dB = 8
        self.bandwidth_Hz = 10e9
        self.noise_temp_K = 290 * 10**(self.noise_figure_dB/10)
        self.N_0 = PhysicalConstants.k * self.noise_temp_K * self.bandwidth_Hz
        
        # Channel gain (linear)
        self.channel_gain = np.sqrt(10**((self.G_tx_dB + self.G_rx_dB - self.path_loss_dB)/10))
        
        # Print enhanced link budget (once)
        if not hasattr(self.__class__, '_link_budget_printed'):
            print(f"\n=== Enhanced THz ISL Link Budget at {self.f_c/1e9:.0f} GHz ===")
            print(f"  Distance: {self.distance/1e3:.0f} km")
            print(f"  Antenna Diameter: {self.antenna_diameter:.1f} m")
            print(f"  Tx Power: {self.P_tx_dBm:.1f} dBm ({self.P_tx_watts*1000:.0f} mW)")
            print(f"  Antenna Gains: {self.G_tx_dB:.1f} dBi each")
            print(f"  Path Loss: {self.path_loss_dB:.1f} dB")
            print(f"  Rx Power: {self.P_rx_dBm:.1f} dBm")
            print(f"  Noise Power: {10*np.log10(self.N_0*1000):.1f} dBm")
            print(f"  Link Margin: {self.P_rx_dBm - 10*np.log10(self.N_0*1000):.1f} dB")
            self.__class__._link_budget_printed = True
    
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
        """Calculate SINR with Monte Carlo pointing error averaging."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        symbol_power = np.abs(symbol)**2
        
        # Monte Carlo averaging over pointing error
        pointing_losses = scenario.sample_pointing_loss(
            self.f_c, self.antenna_diameter, n_samples=n_mc
        )
        
        P_rx_signal_base = P_tx * symbol_power * self.channel_gain**2 * self.bussgang_gain**2
        P_rx_signal_avg = P_rx_signal_base * np.mean(pointing_losses)
        
        N_thermal = self.N_0
        N_hw = P_rx_signal_avg * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        
        N_total = N_thermal + N_hw * phase_penalty
        sinr = P_rx_signal_avg / N_total
        return sinr
    
    def calculate_mutual_information(self, p_x: np.ndarray, P_tx_scale: float = 1.0,
                                   n_mc: int = 100) -> np.ndarray:
        """Calculate mutual information with MC averaging."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr_mc(symbol, avg_power, P_tx_scale, n_mc)
            I_x[i] = np.log2(1 + sinr)
            
        return I_x
    
    def calculate_capacity_vs_snr(self, snr_dB_array: np.ndarray, n_mc: int = 100) -> Dict[str, np.ndarray]:
        """Calculate capacity vs SNR with MC averaging."""
        capacities = []
        
        for snr_dB in snr_dB_array:
            snr_linear = 10**(snr_dB/10)
            
            # MC averaging for pointing error
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
        """Calculate B-FIM with MC averaging."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        
        # MC averaging
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
        
        self._debug_count += 1
        if DEBUG_VERBOSE and self._debug_count % 200 == 0:
            print(f"\n[Debug #{self._debug_count}]")
            print(f"  avg_power = {avg_power:.6f}")
            print(f"  J_B condition = {np.linalg.cond(J_B):.2e}")
        
        try:
            J_B_inv = inv(J_B)
            distortion = J_B_inv[0,0]  # Range variance only
            
            if distortion < 0 or distortion > 1e15:
                return 1e10
                
        except np.linalg.LinAlgError:
            return 1e10
            
        return distortion

# =========================================================================
# ENHANCED PLOT FUNCTIONS
# =========================================================================

def plot_cd_frontier_all_profiles(save_name='fig_cd_frontier_all'):
    """Plot C-D frontier for all hardware profiles."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        print(f"  Processing {profile_name}...")
        system = ISACSystem(profile_name)
        
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
            capacity, p_opt = modified_blahut_arimoto(
                system, D_target, P_tx_scale=1.0, n_mc=50,
                max_iterations=30, verbose=False
            )
            
            actual_D = system.calculate_distortion(p_opt, n_mc=50)
            
            if 0 < actual_D < 1e10 and capacity >= 0:
                distortions.append(actual_D)
                capacities.append(capacity)
        
        if len(distortions) > 0:
            ranging_rmse_mm = np.sqrt(distortions) * 1000
            
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
    ax.set_ylabel('Communication Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('C-D Trade-off for All Hardware Profiles',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    ax.set_xscale('log')
    ax.set_xlim(left=0.01)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def plot_cd_frontier_pointing_sensitivity(save_name='fig_cd_pointing_sensitivity'):
    """Plot C-D frontier sensitivity to pointing error."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profile_name = "High_Performance"
    pointing_errors_urad = [0.5, 1.0, 2.0]  # µrad
    
    for idx, pe_urad in enumerate(pointing_errors_urad):
        print(f"  Processing σ_θ = {pe_urad} µrad...")
        
        # Temporarily override pointing error
        original_pe = scenario.pointing_error_rms_rad
        scenario.pointing_error_rms_rad = pe_urad * 1e-6
        
        system = ISACSystem(profile_name)
        
        # Generate C-D points
        n_points = 10
        distortions = []
        capacities = []
        
        p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
        D_max = system.calculate_distortion(p_uniform, n_mc=50)
        D_min = D_max / 100
        
        D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_points)
        
        for D_target in D_targets:
            capacity, p_opt = modified_blahut_arimoto(
                system, D_target, P_tx_scale=1.0, n_mc=50,
                max_iterations=20, verbose=False
            )
            
            actual_D = system.calculate_distortion(p_opt, n_mc=50)
            
            if 0 < actual_D < 1e10 and capacity >= 0:
                distortions.append(actual_D)
                capacities.append(capacity)
        
        # Restore original
        scenario.pointing_error_rms_rad = original_pe
        
        if len(distortions) > 0:
            ranging_rmse_mm = np.sqrt(distortions) * 1000
            
            ax.plot(ranging_rmse_mm, capacities,
                   color=colors[idx], 
                   linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                   linestyle=linestyles[idx],
                   marker=markers[idx], 
                   markersize=IEEEStyle.LINE_PROPS['markersize'],
                   markerfacecolor='white', 
                   markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                   label=f'σ_θ = {pe_urad} µrad')
    
    # Add performance thresholds
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Labels
    ax.set_xlabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Communication Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Impact of Pointing Error on C-D Trade-off\n(High Performance Hardware)',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    ax.set_xscale('log')
    ax.set_xlim(left=0.01)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def plot_2d_parameter_analysis(save_name='fig_2d_parameter_analysis'):
    """Plot 2D parameter analysis instead of 3D landscape."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Common parameters
    hardware_profile = "High_Performance"
    
    # 1. Capacity vs Frequency for different antenna sizes
    frequencies_GHz = np.linspace(100, 600, 20)
    antenna_sizes = [0.5, 1.0, 1.5, 2.0]
    
    for idx, ant_size in enumerate(antenna_sizes):
        capacities = []
        
        for f_GHz in frequencies_GHz:
            system = ISACSystem(hardware_profile, f_c=f_GHz*1e9, antenna_diameter=ant_size)
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacities.append(np.mean(I_x))
        
        ax1.plot(frequencies_GHz, capacities,
                color=colors[idx], linewidth=2,
                marker=markers[idx], markersize=5, markevery=5,
                label=f'{ant_size} m')
    
    ax1.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_title('(a) Capacity vs. Frequency', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax1.grid(True, **IEEEStyle.GRID_PROPS)
    ax1.legend(title='Antenna Diameter', fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax1.set_ylim(bottom=0)
    
    # 2. Ranging RMSE vs Distance for different frequencies
    distances_km = np.linspace(500, 5000, 20)
    frequencies_GHz = [100, 200, 300, 600]
    
    for idx, f_GHz in enumerate(frequencies_GHz):
        rmse_values = []
        
        for d_km in distances_km:
            system = ISACSystem(hardware_profile, f_c=f_GHz*1e9, distance=d_km*1e3)
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            distortion = system.calculate_distortion(p_x, n_mc=50)
            rmse_mm = np.sqrt(distortion) * 1000
            rmse_values.append(rmse_mm)
        
        ax2.semilogy(distances_km, rmse_values,
                    color=colors[idx], linewidth=2,
                    marker=markers[idx], markersize=5, markevery=5,
                    label=f'{f_GHz} GHz')
    
    ax2.set_xlabel('Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_title('(b) RMSE vs. Distance', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax2.grid(True, **IEEEStyle.GRID_PROPS)
    ax2.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax2.set_ylim(1e-2, 1e3)
    
    # 3. Capacity ceiling vs Gamma_eff for different phase noise
    gamma_eff_range = np.logspace(-3, -1, 30)
    phase_noise_vars = [0.001, 0.01, 0.1]  # rad²
    
    for idx, sigma_phi_sq in enumerate(phase_noise_vars):
        ceilings = []
        
        for gamma_eff in gamma_eff_range:
            ceiling = DerivedParameters.capacity_ceiling(gamma_eff, sigma_phi_sq)
            ceilings.append(ceiling)
        
        ax3.semilogx(gamma_eff_range, ceilings,
                    color=colors[idx], linewidth=2,
                    linestyle=linestyles[idx],
                    label=f'σ_φ² = {sigma_phi_sq} rad²')
    
    # Mark existing hardware
    for name, profile in HARDWARE_PROFILES.items():
        if name != "Custom":
            ax3.axvline(x=profile.Gamma_eff, color='gray', 
                       linestyle=':', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax3.set_ylabel('Capacity Ceiling (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax3.set_title('(c) Hardware Limitations', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax3.grid(True, **IEEEStyle.GRID_PROPS)
    ax3.legend(fontsize=IEEEStyle.FONT_SIZES['legend']-1)
    ax3.set_ylim(0, 8)
    
    # 4. Link margin heatmap
    antenna_sizes = np.linspace(0.3, 2.0, 15)
    tx_powers = np.linspace(10, 33, 15)
    
    link_margins = np.zeros((len(antenna_sizes), len(tx_powers)))
    
    for i, ant_size in enumerate(antenna_sizes):
        for j, tx_power in enumerate(tx_powers):
            ant_gain = scenario.antenna_gain_dB(ant_size, scenario.f_c_default)
            budget = DerivedParameters.link_budget_dB(
                tx_power, ant_gain, ant_gain,
                scenario.R_default, scenario.f_c_default
            )
            noise_dBm = DerivedParameters.thermal_noise_power_dBm(10e9, noise_figure_dB=8)
            link_margins[i, j] = budget['rx_power_dBm'] - noise_dBm
    
    im = ax4.imshow(link_margins.T, cmap='RdYlGn', aspect='auto',
                   extent=[antenna_sizes[0], antenna_sizes[-1],
                          tx_powers[0], tx_powers[-1]],
                   origin='lower', interpolation='bilinear')
    
    CS = ax4.contour(antenna_sizes, tx_powers, link_margins.T, 
                     levels=[0, 5, 10, 15], colors='black', linewidths=1)
    ax4.clabel(CS, inline=True, fontsize=IEEEStyle.FONT_SIZES['annotation']-1)
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Link Margin (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    cbar.ax.tick_params(labelsize=IEEEStyle.FONT_SIZES['tick'])
    
    ax4.set_xlabel('Antenna Diameter (m)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax4.set_ylabel('Transmit Power (dBm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax4.set_title('(d) Link Budget Analysis', fontsize=IEEEStyle.FONT_SIZES['title'])
    
    plt.suptitle('THz ISL ISAC Parameter Analysis', fontsize=IEEEStyle.FONT_SIZES['title']+2)
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def plot_isac_feasibility_regions(save_name='fig_isac_feasibility'):
    """Plot ISAC feasibility regions in parameter space."""
    print(f"\n=== Generating {save_name} ===")
    
    # Parameter ranges
    tx_power_dBm = np.linspace(10, 40, 25)
    distances_km = np.linspace(500, 5000, 25)
    
    # Create meshgrid
    P, D = np.meshgrid(tx_power_dBm, distances_km)
    
    # Define feasibility criteria
    min_capacity = 1.0  # bits/symbol
    max_ranging_rmse = 10.0  # mm
    
    # Calculate feasibility for each point
    comm_feasible = np.zeros_like(P)
    sense_feasible = np.zeros_like(P)
    
    print("  Computing feasibility map...")
    for i in tqdm(range(P.shape[0]), desc="    Distance levels"):
        for j in range(P.shape[1]):
            # Create system with custom power
            system = ISACSystem("SWaP_Efficient", distance=D[i,j]*1e3)
            system.P_tx_dBm = P[i,j]
            system.P_tx_watts = 10**(P[i,j]/10) / 1000
            
            # Recalculate link budget
            system._calculate_enhanced_link_budget()
            
            # Check if link closes
            if system.P_rx_dBm - 10*np.log10(system.N_0*1000) < 0:
                continue
            
            # Check communication feasibility
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacity = np.mean(I_x)
            comm_feasible[i,j] = 1 if capacity >= min_capacity else 0
            
            # Check sensing feasibility
            distortion = system.calculate_distortion(p_x, n_mc=50)
            ranging_rmse = np.sqrt(distortion) * 1000
            sense_feasible[i,j] = 1 if ranging_rmse <= max_ranging_rmse else 0
    
    # Combined feasibility
    isac_feasible = comm_feasible * sense_feasible
    
    # Create plot
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['square'])
    
    # Create custom colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'lightcoral', 'lightblue', 'darkgreen'])
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Combined regions: 0=infeasible, 1=comm only, 2=sense only, 3=both
    combined = comm_feasible + 2*sense_feasible
    
    im = ax.contourf(P, D, combined, levels=bounds, cmap=cmap, norm=norm)
    
    # Add contour lines
    ax.contour(P, D, combined, levels=[0.5, 1.5, 2.5], colors='black', 
               linewidths=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Transmit Power (dBm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('ISL Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('ISAC Feasibility Regions\n(C ≥ 1 bit/symbol, RMSE ≤ 10 mm)', 
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Infeasible'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Communication Only'),
        Patch(facecolor='lightblue', edgecolor='black', label='Sensing Only'),
        Patch(facecolor='darkgreen', edgecolor='black', label='ISAC Feasible')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
             fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: results/{save_name}.pdf and results/{save_name}.png")

def modified_blahut_arimoto(system: ISACSystem, D_target: float, 
                           P_tx_scale: float = 1.0,
                           epsilon_lambda: float = 1e-3,
                           epsilon_p: float = 1e-6,
                           max_iterations: int = 50,
                           n_mc: int = 100,
                           verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Modified Blahut-Arimoto for ISAC with MC averaging."""
    n_symbols = len(system.constellation)
    
    # Smart initialization
    if D_target > 1e6:
        p_x = np.ones(n_symbols) / n_symbols
    else:
        symbol_powers = np.abs(system.constellation)**2
        p_x = symbol_powers / np.sum(symbol_powers)
    
    # Check if uniform distribution already meets target
    D_uniform = system.calculate_distortion(p_x, P_tx_scale, n_mc)
    if D_uniform <= D_target:
        I_x = system.calculate_mutual_information(p_x, P_tx_scale, n_mc)
        return np.sum(p_x * I_x), p_x
    
    # Binary search for Lagrange multiplier
    lambda_min, lambda_max = 0, 1e6
    
    iteration_count = 0
    while (lambda_max - lambda_min) > epsilon_lambda and iteration_count < 20:
        lambda_current = (lambda_min + lambda_max) / 2
        iteration_count += 1
        
        p_x = np.ones(n_symbols) / n_symbols
        
        # Inner optimization
        for inner_iter in range(max_iterations):
            p_x_prev = p_x.copy()
            
            I_x = system.calculate_mutual_information(p_x, P_tx_scale, n_mc)
            
            # Numerical gradient
            grad_D = np.zeros(n_symbols)
            base_D = system.calculate_distortion(p_x, P_tx_scale, n_mc)
            
            delta = 0.01
            for i in range(n_symbols):
                if p_x[i] > delta:
                    p_perturb = p_x.copy()
                    p_perturb[i] -= delta
                    p_perturb[(i+1) % n_symbols] += delta
                    
                    D_perturb = system.calculate_distortion(p_perturb, P_tx_scale, n_mc)
                    grad_D[i] = (D_perturb - base_D) / delta
            
            # Update in log domain
            log_p = np.log(p_x + 1e-10)
            log_p += 0.1 * (I_x - lambda_current * grad_D)
            
            # Normalize
            log_p -= np.max(log_p)
            p_x = np.exp(log_p)
            p_x /= np.sum(p_x)
            
            # Check convergence
            if np.linalg.norm(p_x - p_x_prev, ord=1) < epsilon_p:
                break
        
        # Check constraint
        D_current = system.calculate_distortion(p_x, P_tx_scale, n_mc)
        
        if D_current > D_target:
            lambda_min = lambda_current
        else:
            lambda_max = lambda_current
    
    capacity = np.sum(p_x * I_x)
    return capacity, p_x

def main():
    """Main function with all analyses."""
    print("=== Enhanced THz ISL ISAC Analysis Suite ===")
    print("With pointing error Monte Carlo averaging")
    print("Key Improvements:")
    print("- All hardware profiles included")
    print("- Pointing error sensitivity analysis")
    print("- 2D decomposition of performance")
    print("- Results saved to 'results/' folder")
    
    # Set debug level
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = False
    
    # Generate all plots
    plot_cd_frontier_all_profiles()
    plot_cd_frontier_pointing_sensitivity()
    plot_2d_parameter_analysis()
    plot_isac_feasibility_regions()
    
    print("\n=== Analysis Complete ===")
    print("Generated files in results/:")
    print("- fig_cd_frontier_all.pdf/png")
    print("- fig_cd_pointing_sensitivity.pdf/png")
    print("- fig_2d_parameter_analysis.pdf/png")
    print("- fig_isac_feasibility.pdf/png")

if __name__ == "__main__":
    main()