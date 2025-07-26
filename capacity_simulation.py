#!/usr/bin/env python3
"""
capacity_simulation.py - IEEE Publication Style with Individual Plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
from typing import Tuple, Dict, List
from scipy.linalg import inv
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

class EnhancedISACSystem:
    """Enhanced THz ISL ISAC system with IEEE publication style."""
    
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
    
    def calculate_sinr(self, symbol: complex, avg_power: float, P_tx_scale: float) -> float:
        """Calculate SINR for given symbol and power allocation."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        symbol_power = np.abs(symbol)**2
        P_rx_signal = P_tx * symbol_power * (self.channel_gain**2) * (self.bussgang_gain**2)
        
        N_thermal = self.N_0
        N_hw = P_rx_signal * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        
        N_total = N_thermal + N_hw * phase_penalty
        sinr = P_rx_signal / N_total
        return sinr
    
    def calculate_mutual_information(self, p_x: np.ndarray, P_tx_scale: float = 1.0) -> np.ndarray:
        """Calculate mutual information for each symbol."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr(symbol, avg_power, P_tx_scale)
            I_x[i] = np.log2(1 + sinr)
            
        return I_x
    
    def calculate_capacity_vs_snr(self, snr_dB_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate capacity vs SNR showing hardware ceiling."""
        capacities = []
        
        for snr_dB in snr_dB_array:
            snr_linear = 10**(snr_dB/10)
            
            P_signal = self.P_tx_watts * (self.channel_gain**2) * (self.bussgang_gain**2)
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
    
    def calculate_bfim_observable(self, avg_power: float, P_tx_scale: float) -> np.ndarray:
        """Calculate B-FIM for observable parameters only (2x2 matrix)."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        P_rx = P_tx * (self.channel_gain**2) * (self.bussgang_gain**2)
        
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
    
    def calculate_distortion(self, p_x: np.ndarray, P_tx_scale: float = 1.0) -> float:
        """Calculate sensing distortion for observable parameters."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        if avg_power < 1e-10:
            return 1e10
        
        J_B = self.calculate_bfim_observable(avg_power, P_tx_scale)
        
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
    """Plot C-D frontier for multiple hardware profiles - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        system = EnhancedISACSystem(profile_name)
        
        # Generate C-D points
        n_points = 15
        distortions = []
        capacities = []
        
        # Find distortion range
        p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
        D_max = system.calculate_distortion(p_uniform)
        D_min = D_max / 1000
        
        D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_points)
        
        for D_target in D_targets:
            capacity, p_opt = simple_cd_optimization(system, D_target)
            actual_D = system.calculate_distortion(p_opt)
            
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
                   label=f'{profile_name.replace("_", " ")} ($\Gamma_{{eff}}$={profile.Gamma_eff})')
    
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
    ax.set_title('Capacity-Distortion Trade-off', fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.set_xscale('log')
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_xlim(left=0.01)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}.pdf and {save_name}.png")

def plot_capacity_vs_snr(save_name='fig_capacity_vs_snr'):
    """Plot capacity vs SNR showing hardware ceilings - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    profiles = ["State_of_Art", "High_Performance", "SWaP_Efficient"]
    snr_dB_array = np.linspace(-10, 50, 100)
    
    for idx, profile_name in enumerate(profiles):
        system = EnhancedISACSystem(profile_name)
        results = system.calculate_capacity_vs_snr(snr_dB_array)
        
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
    plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}.pdf and {save_name}.png")

def plot_hardware_quality_impact(save_name='fig_hardware_quality_impact'):
    """Plot capacity vs hardware quality factor - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    gamma_eff_range = np.logspace(-3, -1, 30)
    snr_levels_dB = [10, 20, 30, 40]
    
    for idx, snr_dB in enumerate(snr_levels_dB):
        capacities = []
        
        for gamma_eff in gamma_eff_range:
            custom_profile = HARDWARE_PROFILES["Custom"]
            original_gamma = custom_profile.Gamma_eff
            custom_profile.Gamma_eff = gamma_eff
            
            snr_linear = 10**(snr_dB/10)
            phase_factor = np.exp(-custom_profile.phase_noise_variance)
            sinr_eff = snr_linear / (1 + snr_linear * gamma_eff)
            capacity = np.log2(1 + sinr_eff * phase_factor)
            capacities.append(capacity)
            
            custom_profile.Gamma_eff = original_gamma
        
        ax.semilogx(gamma_eff_range, capacities, 
                   color=colors[idx],
                   linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                   linestyle=linestyles[idx],
                   marker=markers[idx],
                   markersize=IEEEStyle.LINE_PROPS['markersize']-1,
                   markevery=6,
                   markerfacecolor='white',
                   markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                   label=f'SNR = {snr_dB} dB')
    
    # Mark existing hardware
    for name, profile in HARDWARE_PROFILES.items():
        if name != "Custom":
            ax.axvline(x=profile.Gamma_eff, color='gray', 
                      linestyle=':', alpha=0.5, linewidth=1.2)
            ax.text(profile.Gamma_eff*1.1, 7.5, 
                   name.split('_')[0], rotation=90, 
                   fontsize=IEEEStyle.FONT_SIZES['annotation']-1, alpha=0.7)
    
    # Labels
    ax.set_xlabel('Hardware Quality Factor $\Gamma_{eff}$',
                 fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Impact of Hardware Quality on Capacity',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='lower left', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(0, 8)
    ax.set_xlim(1e-3, 1e-1)
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}.pdf and {save_name}.png")

def plot_hardware_feasibility_map(save_name='fig_hardware_feasibility'):
    """Plot 2D hardware feasibility map - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    # Parameter ranges
    gamma_eff_range = np.logspace(-3, -1, 30)
    phase_noise_range = np.logspace(-3, -1, 30)
    
    GAMMA, PHASE = np.meshgrid(gamma_eff_range, phase_noise_range)
    
    # Calculate capacity ceiling
    feasibility = np.zeros_like(GAMMA)
    for i in range(GAMMA.shape[0]):
        for j in range(GAMMA.shape[1]):
            ceiling = DerivedParameters.capacity_ceiling(GAMMA[i,j], PHASE[i,j])
            feasibility[i,j] = ceiling
    
    # Create contour plot
    levels = np.linspace(0, 8, 17)
    cs = ax.contourf(GAMMA, PHASE, feasibility, levels=levels, cmap='viridis')
    
    # Add contour lines for key capacities
    contour_lines = ax.contour(GAMMA, PHASE, feasibility, 
                               levels=[2, 4, 6], colors='white', 
                               linewidths=1.5, alpha=0.8)
    ax.clabel(contour_lines, inline=True, fontsize=IEEEStyle.FONT_SIZES['annotation']-1)
    
    # Mark hardware profiles
    for name, profile in HARDWARE_PROFILES.items():
        if name != "Custom":
            ax.scatter(profile.Gamma_eff, profile.phase_noise_variance,
                      s=100, marker='*', edgecolors='red', 
                      linewidths=2, facecolors='white')
            ax.text(profile.Gamma_eff*1.2, profile.phase_noise_variance,
                   name.split('_')[0], 
                   fontsize=IEEEStyle.FONT_SIZES['annotation']-1)
    
    # Labels
    ax.set_xlabel('Hardware Quality Factor $\Gamma_{eff}$',
                 fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Phase Noise Variance $\sigma^2_{\phi}$ (rad$^2$)',
                 fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Hardware Feasibility Map',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Capacity Ceiling (bits/symbol)', 
                  fontsize=IEEEStyle.FONT_SIZES['label'])
    cbar.ax.tick_params(labelsize=IEEEStyle.FONT_SIZES['tick'])
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}.pdf and {save_name}.png")

def plot_antenna_power_tradeoff(save_name='fig_antenna_power_tradeoff'):
    """Plot antenna size vs power trade-off - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['large'])
    
    antenna_sizes = np.array([0.3, 0.5, 1.0, 1.5, 2.0])
    tx_powers = np.array([10, 20, 30, 33])
    
    # Calculate link margin for each combination
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
    
    # Create heatmap
    im = ax.imshow(link_margins, cmap='RdYlGn', aspect='auto',
                  extent=[tx_powers[0]-2.5, tx_powers[-1]+2.5,
                         antenna_sizes[0]-0.15, antenna_sizes[-1]+0.15],
                  origin='lower', interpolation='bilinear')
    
    # Add contour lines
    X, Y = np.meshgrid(tx_powers, antenna_sizes)
    CS = ax.contour(X, Y, link_margins, levels=[0, 5, 10, 15], 
                   colors='black', linewidths=1.5)
    ax.clabel(CS, inline=True, fontsize=IEEEStyle.FONT_SIZES['annotation'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Link Margin (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    cbar.ax.tick_params(labelsize=IEEEStyle.FONT_SIZES['tick'])
    
    # Mark recommended region
    rect = plt.Rectangle((25, 0.8), 8, 1.2, fill=False, 
                        edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(29, 1.4, 'Recommended', ha='center', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='blue')
    
    # Labels
    ax.set_xlabel('Transmit Power (dBm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Antenna Diameter (m)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Link Budget Trade-off Analysis',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}.pdf and {save_name}.png")

def plot_performance_summary(save_name='fig_performance_summary'):
    """Plot performance summary comparison - IEEE style."""
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
        
        rmse = 1000 * np.sqrt(1 / (snr_linear * (f_c/3e8)**2 * profile.Gamma_eff**(-1)))
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
    
    # Add value labels
    for bars, values in [(bars1, ranging_rmse), (bars2, capacity), (bars3, hw_limit_snr)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if bars == bars1:
                label = f'{val:.1f}\nmm'
            elif bars == bars2:
                label = f'{val:.1f}\nb/s'
            else:
                label = f'{val:.0f}\ndB'
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   label, ha='center', va='bottom', 
                   fontsize=IEEEStyle.FONT_SIZES['annotation']-1)
    
    # Labels
    ax.set_xlabel('Hardware Profile', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Normalized Performance', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Performance Summary at SNR = 20 dB',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in profiles], 
                      fontsize=IEEEStyle.FONT_SIZES['tick'])
    ax.legend(loc='upper left', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(0, 1.3)
    ax.grid(True, axis='y', **IEEEStyle.GRID_PROPS)
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}.pdf and {save_name}.png")

def simple_cd_optimization(system: EnhancedISACSystem, D_target: float,
                          max_iterations: int = 20) -> Tuple[float, np.ndarray]:
    """Simplified C-D optimization for faster computation."""
    n_symbols = len(system.constellation)
    
    # Initialize with uniform distribution
    p_x = np.ones(n_symbols) / n_symbols
    
    # Check if already meets constraint
    D_current = system.calculate_distortion(p_x)
    if D_current <= D_target:
        I_x = system.calculate_mutual_information(p_x)
        return np.sum(p_x * I_x), p_x
    
    # Simple gradient-based optimization
    step_size = 0.1
    for _ in range(max_iterations):
        # Calculate gradient
        I_x = system.calculate_mutual_information(p_x)
        
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
        D_current = system.calculate_distortion(p_x)
    
    capacity = np.sum(p_x * I_x)
    return capacity, p_x

def main():
    """Main function to generate all capacity analysis plots."""
    print("=== THz ISL ISAC Capacity Analysis (IEEE Style) ===")
    print("\nGenerating individual plots with adjustable font sizes...")
    print(f"Current font sizes: {IEEEStyle.FONT_SIZES}")
    
    # Note about observability
    print("\nNOTE: Analysis based on single ISL (2 observable parameters)")
    
    # Generate all individual plots
    plot_cd_frontier()
    plot_capacity_vs_snr()
    plot_hardware_quality_impact()
    plot_hardware_feasibility_map()
    plot_antenna_power_tradeoff()
    plot_performance_summary()
    
    print("\n=== Capacity Analysis Complete ===")
    print("Generated files:")
    print("- fig_cd_frontier.pdf/png")
    print("- fig_capacity_vs_snr.pdf/png")
    print("- fig_hardware_quality_impact.pdf/png")
    print("- fig_hardware_feasibility.pdf/png")
    print("- fig_antenna_power_tradeoff.pdf/png")
    print("- fig_performance_summary.pdf/png")

if __name__ == "__main__":
    main()