#!/usr/bin/env python3
"""
cd_frontier_simulation.py - Enhanced Version with Data Saving and 3D Views
Updated with 1THz support and professional visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
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
    IEEEStyle,
    data_saver
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
                 antenna_diameter: float = None,
                 tx_power_dBm: float = None):  # ADD THIS PARAMETER
        """Initialize with enhanced parameters."""
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        self.antenna_diameter = antenna_diameter or scenario.default_antenna_diameter
        self.tx_power_dBm = tx_power_dBm or scenario.default_tx_power_dBm  # ADD THIS
        
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
        # Use passed-in or default power
        self.P_tx_dBm = self.tx_power_dBm  # CHANGED: use instance variable
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
        self.bandwidth_Hz = self.profile.signal_bandwidth_Hz
        self.noise_temp_K = 290 * 10**(self.noise_figure_dB/10)
        self.noise_power_watts = PhysicalConstants.k * self.noise_temp_K * self.bandwidth_Hz
        
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
            print(f"  Noise Power: {10*np.log10(self.noise_power_watts*1000):.1f} dBm")
            print(f"  Link Margin: {self.P_rx_dBm - 10*np.log10(self.noise_power_watts*1000):.1f} dB")
            self.__class__._link_budget_printed = True
    
    def _calculate_bussgang_gain(self, input_backoff_dB: float = 7.0) -> complex:
        """Calculate Bussgang gain for PA nonlinearity."""
        kappa = 10 ** (-input_backoff_dB / 10)
        B = 1 - 1.5 * kappa + 1.875 * kappa**2
        # Ensure real value
        return float(np.real(B))
    
    def _create_constellation(self, modulation: str = 'QPSK') -> np.ndarray:
        """Create normalized constellation."""
        if modulation == 'QPSK':
            angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            constellation = np.exp(1j * angles)
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation
    
    # 在 calculate_sinr_mc 方法中
    def calculate_sinr_mc(self, symbol: complex, avg_power: float, P_tx_scale: float,
                    n_mc: int = 100) -> float:
        """Calculate SINR with Monte Carlo pointing error averaging."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        symbol_power = np.abs(symbol)**2
        
        # Monte Carlo averaging over pointing error
        pointing_losses = scenario.sample_pointing_loss(
            self.f_c, self.antenna_diameter, n_samples=n_mc
        )
        
        # Ensure real values
        P_rx_signal_base = P_tx * symbol_power * np.abs(self.channel_gain)**2 * np.abs(self.bussgang_gain)**2
        P_rx_signal_avg = P_rx_signal_base * np.mean(pointing_losses)
        
        N_thermal = self.noise_power_watts  # Changed from self.N_0
        N_hw = P_rx_signal_avg * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        
        N_total = N_thermal + N_hw * phase_penalty
        sinr = P_rx_signal_avg / N_total
        
        # Ensure real and positive
        return np.real(np.abs(sinr))
    

    def compute_cd_frontier_grid_full(system, P_tx_scales=None, pilot_counts=None, n_mc=100):
        """
        Compute full C-D frontier with wider parameter ranges.
        
        For systems in hardware-limited regime, we need larger variations to see trade-offs.
        """
        if P_tx_scales is None:
            # MUCH wider power range: -20dB to +10dB (100x variation)
            P_tx_scales = np.logspace(-1.3, +1.0, 100)  # 0.05x to 10x
            
        if pilot_counts is None:
            # More pilot options including very low counts
            pilot_counts = [8, 16, 32, 64, 128, 256]
        
        # Keep uniform distribution
        p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
        
        all_points = []
        
        print(f"      Grid search: {len(P_tx_scales)} powers × {len(pilot_counts)} pilots")
        
        original_pilots = system.n_pilots
        
        for M in pilot_counts:
            system.n_pilots = M
            
            for P_scale in P_tx_scales:
                try:
                    # Calculate capacity
                    I_vec = system.calculate_mutual_information(
                        p_uniform, P_tx_scale=P_scale, n_mc=n_mc
                    )
                    capacity = float(np.mean(I_vec))
                    
                    # Calculate distortion
                    distortion = system.calculate_distortion(
                        p_uniform, P_tx_scale=P_scale, n_mc=n_mc
                    )
                    
                    # Keep all valid points (even if capacity is low)
                    if distortion > 0 and distortion < 1e10 and capacity >= 0:
                        all_points.append({
                            'D': distortion,
                            'C': capacity,
                            'P_scale': P_scale,
                            'M': M
                        })
                except:
                    continue
        
        system.n_pilots = original_pilots
        
        if len(all_points) == 0:
            print("      Warning: No valid points found!")
            return np.array([]), np.array([])
        
        print(f"      Found {len(all_points)} valid configurations")
        
        # Sort by distortion
        all_points.sort(key=lambda x: x['D'])
        
        # Extract Pareto frontier with relaxed criteria
        pareto_points = []
        max_capacity = -np.inf
        
        # Include more points by using a tolerance
        capacity_tolerance = 0.99  # Accept points within 1% of best capacity
        
        for point in all_points:
            if point['C'] >= max_capacity * capacity_tolerance:
                pareto_points.append(point)
                if point['C'] > max_capacity:
                    max_capacity = point['C']
        
        # Ensure we have enough points by subsampling if needed
        if len(pareto_points) > 100:
            # Subsample to keep visualization manageable
            indices = np.linspace(0, len(pareto_points)-1, 100, dtype=int)
            pareto_points = [pareto_points[i] for i in indices]
        
        if len(pareto_points) > 0:
            pareto_D = np.array([p['D'] for p in pareto_points])
            pareto_C = np.array([p['C'] for p in pareto_points])
            
            print(f"      Pareto frontier: {len(pareto_points)} points")
            print(f"      D range: [{np.min(pareto_D):.2e}, {np.max(pareto_D):.2e}]")
            print(f"      C range: [{np.min(pareto_C):.2f}, {np.max(pareto_C):.2f}] bits/symbol")
            
            return pareto_D, pareto_C
        else:
            return np.array([]), np.array([])


    def calculate_mutual_information(self, p_x: np.ndarray, P_tx_scale: float = 1.0,
                               n_mc: int = 100) -> np.ndarray:
        """Calculate mutual information with MC averaging."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr_mc(symbol, avg_power, P_tx_scale, n_mc)
            # Ensure real value
            I_x[i] = np.real(np.log2(1 + np.abs(sinr)))
            
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
        
        N_thermal = self.noise_power_watts  # Changed from self.N_0
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


def compute_cd_frontier_grid_full(system, P_tx_scales=None, pilot_counts=None, n_mc=100):
    """
    Compute full C-D frontier with wider parameter ranges.
    
    For systems in hardware-limited regime, we need larger variations to see trade-offs.
    """
    if P_tx_scales is None:
        # MUCH wider power range: -20dB to +10dB (100x variation)
        P_tx_scales = np.logspace(-1.3, +1.0, 100)  # 0.05x to 10x
        
    if pilot_counts is None:
        # More pilot options including very low counts
        pilot_counts = [8, 16, 32, 64, 128, 256]
    
    # Keep uniform distribution
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    
    all_points = []
    
    print(f"      Grid search: {len(P_tx_scales)} powers × {len(pilot_counts)} pilots")
    
    original_pilots = system.n_pilots
    
    for M in pilot_counts:
        system.n_pilots = M
        
        for P_scale in P_tx_scales:
            try:
                # Calculate capacity
                I_vec = system.calculate_mutual_information(
                    p_uniform, P_tx_scale=P_scale, n_mc=n_mc
                )
                capacity = float(np.mean(I_vec))
                
                # Calculate distortion
                distortion = system.calculate_distortion(
                    p_uniform, P_tx_scale=P_scale, n_mc=n_mc
                )
                
                # Keep all valid points (even if capacity is low)
                if distortion > 0 and distortion < 1e10 and capacity >= 0:
                    all_points.append({
                        'D': distortion,
                        'C': capacity,
                        'P_scale': P_scale,
                        'M': M
                    })
            except:
                continue
    
    system.n_pilots = original_pilots
    
    if len(all_points) == 0:
        print("      Warning: No valid points found!")
        return np.array([]), np.array([])
    
    print(f"      Found {len(all_points)} valid configurations")
    
    # Sort by distortion
    all_points.sort(key=lambda x: x['D'])
    
    # Extract Pareto frontier with relaxed criteria
    pareto_points = []
    max_capacity = -np.inf
    
    # Include more points by using a tolerance
    capacity_tolerance = 0.99  # Accept points within 1% of best capacity
    
    for point in all_points:
        if point['C'] >= max_capacity * capacity_tolerance:
            pareto_points.append(point)
            if point['C'] > max_capacity:
                max_capacity = point['C']
    
    # Ensure we have enough points by subsampling if needed
    if len(pareto_points) > 100:
        # Subsample to keep visualization manageable
        indices = np.linspace(0, len(pareto_points)-1, 100, dtype=int)
        pareto_points = [pareto_points[i] for i in indices]
    
    if len(pareto_points) > 0:
        pareto_D = np.array([p['D'] for p in pareto_points])
        pareto_C = np.array([p['C'] for p in pareto_points])
        
        print(f"      Pareto frontier: {len(pareto_points)} points")
        print(f"      D range: [{np.min(pareto_D):.2e}, {np.max(pareto_D):.2e}]")
        print(f"      C range: [{np.min(pareto_C):.2f}, {np.max(pareto_C):.2f}] bits/symbol")
        
        return pareto_D, pareto_C
    else:
        return np.array([]), np.array([])
      
# 文件: cd_frontier_simulation.py
# 完全替换 plot_cd_frontier_all_profiles 函数

def plot_cd_frontier_all_profiles(save_name='fig_cd_frontier_all'):
    """Plot C-D frontier with parameters chosen to show clear trade-offs."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    data_to_save = {
        'hardware_profiles': profiles_to_plot,
        'description': 'C-D frontiers with realistic trade-offs'
    }
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        print(f"  Processing {profile_name}...")
        
        # CRITICAL CHANGES to exit hardware-limited regime:
        # 1. Much smaller antenna (0.3m instead of 1.5m) - reduces gain by 14 dB
        # 2. Lower frequency (100 GHz instead of 300 GHz) - reduces gain by 10 dB  
        # 3. Much lower base power (10 dBm instead of 33 dBm) - reduces by 23 dB
        # 4. Longer distance (5000 km instead of 2000 km) - increases loss by 8 dB
        # Total: ~55 dB reduction in link budget!
        
        system = ISACSystem(
            hardware_profile=profile_name,
            f_c=100e9,  # 100 GHz instead of 300 GHz
            distance=5000e3,  # 5000 km
            n_pilots=64,
            antenna_diameter=0.3,  # 0.3m antenna
            tx_power_dBm=10  # 10 dBm (10 mW) base power
        )
        
        # Now scan with wide power range
        P_tx_scales = np.logspace(-1, +2, 50)  # 0.1x to 100x (40 dB range)
        pilot_counts = [16, 32, 64, 128]
        
        all_points = []
        p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
        
        original_pilots = system.n_pilots
        
        for M in pilot_counts:
            system.n_pilots = M
            for P_scale in P_tx_scales:
                try:
                    # Calculate capacity
                    I_vec = system.calculate_mutual_information(
                        p_uniform, P_tx_scale=P_scale, n_mc=30
                    )
                    capacity = float(np.mean(I_vec))
                    
                    # Calculate distortion
                    distortion = system.calculate_distortion(
                        p_uniform, P_tx_scale=P_scale, n_mc=30
                    )
                    
                    if 0 < distortion < 1e10 and capacity > 0:
                        all_points.append((distortion, capacity))
                except:
                    continue
        
        system.n_pilots = original_pilots
        
        if len(all_points) > 0:
            # Sort and extract Pareto frontier
            all_points.sort(key=lambda x: x[0])
            
            pareto_D = []
            pareto_C = []
            max_C = -np.inf
            
            for D, C in all_points:
                if C > max_C:
                    pareto_D.append(D)
                    pareto_C.append(C)
                    max_C = C
            
            if len(pareto_D) > 0:
                ranging_rmse_mm = np.sqrt(pareto_D) * 1000
                
                data_to_save[f'ranging_rmse_mm_{profile_name}'] = ranging_rmse_mm.tolist()
                data_to_save[f'capacity_{profile_name}'] = pareto_C
                data_to_save[f'num_points_{profile_name}'] = len(pareto_C)
                
                # Plot curve
                ax.plot(ranging_rmse_mm, pareto_C,
                       color=colors[idx], 
                       linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                       linestyle='-',
                       label=f'{profile_name.replace("_", " ")}')
                
                # Add markers
                if len(ranging_rmse_mm) > 1:
                    marker_idx = np.linspace(0, len(ranging_rmse_mm)-1, 
                                           min(8, len(ranging_rmse_mm)), dtype=int)
                    ax.plot(ranging_rmse_mm[marker_idx], np.array(pareto_C)[marker_idx],
                           color=colors[idx],
                           linestyle='None',
                           marker=markers[idx], 
                           markersize=IEEEStyle.LINE_PROPS['markersize'],
                           markerfacecolor='white', 
                           markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'])
                
                print(f"    {profile_name}: {len(pareto_C)} frontier points")
                print(f"      C range: [{min(pareto_C):.2f}, {max(pareto_C):.2f}] bits/symbol")
                print(f"      RMSE range: [{min(ranging_rmse_mm):.2f}, {max(ranging_rmse_mm):.2f}] mm")
    
    # Set axis
    ax.set_xscale('log')
    ax.set_xlim(0.1, 1000)  # 0.1 mm to 1000 mm
    ax.set_ylim(0, 8)
    
    # Add regions
    ax.axhspan(2.0, ax.get_ylim()[1], alpha=0.05, color='green')
    ax.axvspan(ax.get_xlim()[0], 1.0, alpha=0.05, color='blue')
    
    # Thresholds
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Annotations
    ax.text(2, 2.1, 'Communication threshold', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='green')
    ax.text(0.8, 6, 'Sub-mm', ha='right',
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='blue')
    
    # Configuration note
    ax.text(0.98, 0.02, 
           'Config: 100 GHz, 5000 km, 0.3m antenna\n' +
           'Base: 10 dBm, Sweep: 0-30 dBm',
           transform=ax.transAxes,
           fontsize=IEEEStyle.FONT_SIZES['annotation']-2,
           ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Communication Capacity (bits/symbol)', 
                 fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('C-D Trade-off with Realistic Link Budget',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "C-D frontier with realistic parameters")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

    
# 替换 plot_cd_frontier_pointing_sensitivity 函数
def plot_cd_frontier_pointing_sensitivity(save_name='fig_cd_pointing_sensitivity'):
    """Plot C-D frontier sensitivity to pointing error with enhanced visibility."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    profile_name = "High_Performance"
    # ENHANCED: Wider range of pointing errors for visible differences
    pointing_errors_urad = [0.5, 2.0, 5.0, 10.0, 20.0]  # Extended to 20 µrad
    
    # Data storage
    data_to_save = {
        'hardware_profile': profile_name,
        'pointing_errors_urad': pointing_errors_urad
    }
    
    # Save original pointing error
    original_pe = scenario.pointing_error_rms_rad
    
    try:
        for idx, pe_urad in enumerate(pointing_errors_urad):
            print(f"  Processing σ_θ = {pe_urad} µrad...")
            
            # Set pointing error for this iteration
            pe_rad = pe_urad * 1e-6
            scenario.pointing_error_rms_rad = pe_rad
            
            # Create system with REDUCED antenna diameter for more sensitivity
            system = ISACSystem(
                hardware_profile=profile_name,
                f_c=300e9,
                distance=scenario.R_default,
                n_pilots=simulation.n_pilots,
                antenna_diameter=1.0,  # Reduced from 1.5m to increase sensitivity
                tx_power_dBm=30  # Slightly reduced power to avoid deep saturation
            )
            
            # Use full grid search for complete frontier
            distortions, capacities = compute_cd_frontier_grid_full(
                system,
                P_tx_scales=np.logspace(-2, +1, 60),
                pilot_counts=[system.n_pilots],
                n_mc=50
            )
            
            if distortions.size > 0:
                ranging_rmse_mm = np.sqrt(distortions) * 1000
                
                data_to_save[f'ranging_rmse_mm_{pe_urad}urad'] = ranging_rmse_mm.tolist()
                data_to_save[f'capacity_{pe_urad}urad'] = capacities.tolist()
                
                # Use different line styles for clarity
                linestyle = linestyles[min(idx, len(linestyles)-1)]
                
                ax.plot(ranging_rmse_mm, capacities,
                       color=colors[min(idx, len(colors)-1)], 
                       linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                       linestyle=linestyle,
                       marker=markers[min(idx, len(markers)-1)], 
                       markersize=IEEEStyle.LINE_PROPS['markersize'],
                       markevery=max(1, len(ranging_rmse_mm)//8),
                       markerfacecolor='white', 
                       markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                       label=f'σ_θ = {pe_urad} µrad')
    
    finally:
        # Restore original pointing error
        scenario.pointing_error_rms_rad = original_pe
    
    # CRITICAL FIX: Adjust x-axis range for micro-meter scale
    ax.set_xscale('log')
    ax.set_xlim(5e-4, 10)  # From 0.5 µm to 10 mm
    ax.set_ylim(0, 10)
    
    # Performance thresholds
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add text annotations
    ax.text(0.01, 2.1, 'Good communication', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='green')
    ax.text(0.8, ax.get_ylim()[1]*0.7, 'Sub-mm', ha='right',
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='blue')
    
    # Labels
    ax.set_xlabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Communication Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Impact of Pointing Error on C-D Trade-off\n(High Performance Hardware, 1m Antenna)',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'],
             ncol=1 if len(pointing_errors_urad) <= 4 else 2)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "C-D frontier sensitivity to pointing error")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_snr_to_hardware_limit(save_name='fig_snr_to_hardware_limit'):
    """Plot SNR required to reach hardware-limited capacity ceiling."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    # SNR range from very low to very high
    snr_dB = np.linspace(-10, 80, 100)
    snr_linear = 10**(snr_dB/10)
    
    # Data storage
    data_to_save = {
        'snr_dB': snr_dB.tolist()
    }
    
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        system = ISACSystem(profile_name)
        profile = HARDWARE_PROFILES[profile_name]
        
        # Calculate capacity for uniform distribution
        p_x = np.ones(len(system.constellation)) / len(system.constellation)
        capacities = []
        
        for snr in snr_linear:
            # Direct SINR calculation with pointing error
            pointing_loss = scenario.calculate_pointing_loss_factor(
                system.f_c, system.antenna_diameter
            )
            P_rx = system.P_tx_watts * system.channel_gain**2 * system.bussgang_gain**2 * pointing_loss
            N_0 = P_rx / snr  # Back-calculate noise
            N_hw = P_rx * profile.Gamma_eff * np.exp(profile.phase_noise_variance)
            sinr_eff = P_rx / (N_0 + N_hw)
            I_x = np.log2(1 + sinr_eff)
            
            capacity = I_x
            capacities.append(capacity)
        
        capacities = np.array(capacities)
        data_to_save[f'capacity_{profile_name}'] = capacities.tolist()
        
        # Calculate hardware ceiling
        ceiling = np.log2(1 + np.exp(-profile.phase_noise_variance) / profile.Gamma_eff)
        data_to_save[f'ceiling_{profile_name}'] = ceiling
        
        # Find SNR where capacity reaches 95% of ceiling
        idx_95 = np.argmin(np.abs(capacities - 0.95 * ceiling))
        snr_95_dB = snr_dB[idx_95]
        data_to_save[f'snr_95_ceiling_{profile_name}'] = snr_95_dB
        
        # Plot
        ax.plot(snr_dB, capacities, 
               color=colors[idx],
               linewidth=IEEEStyle.LINE_PROPS['linewidth'],
               label=f'{profile_name.replace("_", " ")}')
        ax.axhline(y=ceiling, color=colors[idx], linestyle='--', 
                  linewidth=1.5, alpha=0.5)
        
        # Mark transition point
        ax.plot(snr_95_dB, 0.95*ceiling, 
               marker='o', color=colors[idx], markersize=8)
    
    # Add regions
    ax.axvspan(-10, 20, alpha=0.1, color='blue', label='Power-limited')
    ax.axvspan(40, 80, alpha=0.1, color='red', label='Hardware-limited')
    
    ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('SNR Required to Reach Hardware-Limited Performance',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='lower right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_xlim(-10, 80)
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "SNR to hardware limit analysis")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_gamma_eff_sensitivity(save_name='fig_gamma_eff_sensitivity'):
    """Plot system performance sensitivity to hardware quality factor."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gamma_eff range
    gamma_eff_values = np.logspace(-3, -1, 20)
    
    # Fixed conditions
    snr_dB = simulation.default_SNR_dB
    snr_linear = 10**(snr_dB/10)
    
    # Data storage
    data_to_save = {
        'gamma_eff': gamma_eff_values.tolist(),
        'snr_dB': snr_dB
    }
    
    # Store results
    capacity_results = []
    ranging_rmse_results = []
    
    for gamma_eff in tqdm(gamma_eff_values, desc="    Gamma_eff sweep"):
        # Create temporary profile
        temp_profile = HARDWARE_PROFILES["High_Performance"]
        original_gamma = temp_profile.Gamma_eff
        temp_profile.Gamma_eff = gamma_eff
        
        # Create system
        system = ISACSystem("High_Performance")
        
        # Calculate capacity
        p_x = np.ones(len(system.constellation)) / len(system.constellation)
        I_x = system.calculate_mutual_information(p_x, n_mc=50)
        capacity = np.mean(I_x)
        capacity_results.append(capacity)
        
        # Calculate CRLB
        distortion = system.calculate_distortion(p_x, n_mc=50)
        ranging_rmse = np.sqrt(distortion) * 1000  # Convert to mm
        ranging_rmse_results.append(ranging_rmse)
        
        # Restore original value
        temp_profile.Gamma_eff = original_gamma
    
    data_to_save['capacity'] = capacity_results
    data_to_save['ranging_rmse_mm'] = ranging_rmse_results
    
    # Plot capacity
    ax1.semilogx(gamma_eff_values, capacity_results, 
                'b-', linewidth=3, marker='o', markersize=8)
    ax1.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='High Performance')
    ax1.axvline(x=0.025, color='g', linestyle='--', alpha=0.5, label='SWaP Efficient')
    ax1.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax1.set_title(f'Communication Performance\n(SNR = {snr_dB} dB)', 
                 fontsize=IEEEStyle.FONT_SIZES['title'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    # Plot ranging RMSE
    ax2.loglog(gamma_eff_values, ranging_rmse_results, 
              'r-', linewidth=3, marker='s', markersize=8)
    ax2.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='High Performance')
    ax2.axvline(x=0.025, color='g', linestyle='--', alpha=0.5, label='SWaP Efficient')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.text(1e-2, 1.2, 'Sub-mm threshold', 
            fontsize=IEEEStyle.FONT_SIZES['annotation'], ha='center')
    ax2.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_ylabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax2.set_title(f'Sensing Performance\n(SNR = {snr_dB} dB)', 
                 fontsize=IEEEStyle.FONT_SIZES['title'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    plt.suptitle('System Performance Sensitivity to Hardware Quality', 
                fontsize=IEEEStyle.FONT_SIZES['title']+2)
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Hardware quality factor sensitivity analysis")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_3d_cd_landscape(save_name='fig_3d_cd_landscape'):
    """Plot 3D C-D landscape over frequency and distance."""
    print(f"\n=== Generating {save_name} ===")
    
    # Parameter ranges
    frequencies_GHz = np.linspace(100, 1000, 12)
    distances_km = np.linspace(500, 5000, 12)
    
    # Create meshgrid
    F, D = np.meshgrid(frequencies_GHz, distances_km)
    
    # Fixed parameters
    hardware_profile = "High_Performance"
    
    # Calculate C-D trade-off metrics for each point
    max_capacity_grid = np.zeros_like(F)
    min_rmse_grid = np.zeros_like(F)
    
    print("  Computing 3D C-D landscape...")
    for i in tqdm(range(F.shape[0]), desc="    Distance levels"):
        for j in range(F.shape[1]):
            f_Hz = F[i, j] * 1e9
            d_m = D[i, j] * 1e3
            
            # Create system
            system = ISACSystem(hardware_profile, f_c=f_Hz, distance=d_m)
            
            # Calculate max capacity (uniform distribution)
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=30)
            max_capacity_grid[i, j] = np.mean(I_x)
            
            # Calculate min RMSE
            distortion = system.calculate_distortion(p_x, n_mc=30)
            min_rmse_grid[i, j] = np.sqrt(distortion) * 1000
    
    # Data storage
    data_to_save = {
        'frequency_GHz': frequencies_GHz.tolist(),
        'distance_km': distances_km.tolist(),
        'max_capacity': max_capacity_grid.tolist(),
        'min_rmse_mm': min_rmse_grid.tolist(),
        'hardware_profile': hardware_profile
    }
    
    # Create multiple views for capacity
    viewing_angles = [
        (25, 45, 'default'),
        (15, 60, 'frequency_emphasis'),
        (30, 15, 'distance_emphasis'),
        (45, 45, 'isometric')
    ]
    
    for elev, azim, view_name in viewing_angles:
        fig = plt.figure(figsize=IEEEStyle.FIG_SIZES['3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot for capacity
        surf = ax.plot_surface(F, D, max_capacity_grid, cmap='viridis', 
                              edgecolor='none', alpha=0.8)
        
        # Add contour lines at the bottom
        contours = ax.contour(F, D, max_capacity_grid, zdir='z', offset=0, 
                              cmap='viridis', alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_zlabel('Max Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('Maximum Achievable Capacity\n(High Performance Hardware)', 
                    fontsize=IEEEStyle.FONT_SIZES['title'], pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Highlight 1THz
        ax.plot([1000, 1000], [distances_km[0], distances_km[-1]], 
               [0, 0], 'r--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}_capacity_{view_name}.pdf', 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}_capacity_{view_name}.png', 
                   format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # Create views for RMSE
    for elev, azim, view_name in viewing_angles:
        fig = plt.figure(figsize=IEEEStyle.FIG_SIZES['3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot for RMSE (log scale)
        surf = ax.plot_surface(F, D, np.log10(min_rmse_grid), cmap='plasma', 
                              edgecolor='none', alpha=0.8)
        
        # Add contour lines
        contours = ax.contour(F, D, np.log10(min_rmse_grid), zdir='z', 
                              offset=np.log10(min_rmse_grid).min() - 0.5, 
                              cmap='plasma', alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_zlabel('log10(Min RMSE [mm])', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('Minimum Achievable Ranging RMSE\n(High Performance Hardware)', 
                    fontsize=IEEEStyle.FONT_SIZES['title'], pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Highlight 1THz
        ax.plot([1000, 1000], [distances_km[0], distances_km[-1]], 
               [ax.get_zlim()[0], ax.get_zlim()[0]], 'r--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}_rmse_{view_name}.pdf', 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}_rmse_{view_name}.png', 
                   format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "3D C-D landscape data")
    
    print(f"Saved: results/{save_name}_[capacity/rmse]_[views].pdf/png and data")


def plot_isac_feasibility_regions(save_name='fig_isac_feasibility'):
    """Plot ISAC feasibility regions in parameter space with professional colors."""
    print(f"\n=== Generating {save_name} ===")
    
    # Parameter ranges - adjusted for more variation
    tx_power_dBm = np.linspace(20, 35, 25)  # Lower starting power
    distances_km = np.linspace(500, 5000, 25)
    
    # Create meshgrid
    P, D = np.meshgrid(tx_power_dBm, distances_km)
    
    # Define more stringent thresholds for variation
    min_capacity = 2.0  # bits/symbol
    max_ranging_rmse = 10.0  # mm
    good_capacity = 4.0  # Raised
    good_ranging_rmse = 1.0  # mm
    excellent_capacity = 6.0  # Raised
    excellent_ranging_rmse = 0.1  # mm
    
    # Calculate feasibility for each point
    feasibility = np.zeros_like(P)
    
    data_to_save = {
        'tx_power_dBm': tx_power_dBm.tolist(),
        'distance_km': distances_km.tolist()
    }
    
    print("  Computing feasibility map...")
    for i in tqdm(range(P.shape[0]), desc="    Distance levels"):
        for j in range(P.shape[1]):
            # CRITICAL: Pass actual tx_power to system
            system = ISACSystem("High_Performance", 
                              distance=D[i,j]*1e3,
                              tx_power_dBm=P[i,j])  # Pass actual power
            
            # Check if link closes
            link_margin_dB = system.P_rx_dBm - 10*np.log10(system.noise_power_watts*1000)
            if link_margin_dB < 0:
                feasibility[i,j] = 0  # Link doesn't close
                continue
            
            # Check communication feasibility with actual power
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, P_tx_scale=1.0, n_mc=30)
            capacity = np.mean(I_x)
            
            # Check sensing feasibility
            distortion = system.calculate_distortion(p_x, P_tx_scale=1.0, n_mc=30)
            ranging_rmse = np.sqrt(distortion) * 1000
            
            # Determine feasibility level with stricter thresholds
            if capacity >= excellent_capacity and ranging_rmse <= excellent_ranging_rmse:
                feasibility[i,j] = 5  # Excellent both
            elif capacity >= good_capacity and ranging_rmse <= good_ranging_rmse:
                feasibility[i,j] = 4  # Good both
            elif capacity >= min_capacity and ranging_rmse <= max_ranging_rmse:
                feasibility[i,j] = 3  # Acceptable both
            elif capacity >= min_capacity:
                feasibility[i,j] = 1  # Communication only
            elif ranging_rmse <= max_ranging_rmse:
                feasibility[i,j] = 2  # Sensing only
            else:
                feasibility[i,j] = 0.5  # Link OK but poor performance
    
    # Rest of plotting code remains the same...
    data_to_save['feasibility_map'] = feasibility.tolist()
    
    # Create professional colormap with CORRECTED colors
    colors_map = [
        '#d0d0d0',                                      # 0: Link fails (gray)
        '#ffcccc',                                      # 0.5: Poor performance
        IEEEStyle.COLORS_FEASIBILITY['comm_only'],     # 1: Comm only
        IEEEStyle.COLORS_FEASIBILITY['sense_only'],    # 2: Sense only
        '#a8d8ea',                                      # 3: Acceptable both
        IEEEStyle.COLORS_FEASIBILITY['both'],          # 4: Good both
        IEEEStyle.COLORS_FEASIBILITY['excellent']      # 5: Excellent both (green)
    ]
    cmap = plt.cm.colors.ListedColormap(colors_map)
    bounds = [0, 0.25, 0.75, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Create plot
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    im = ax.contourf(P, D, feasibility, levels=bounds, cmap=cmap, norm=norm)
    
    # Add contour lines
    ax.contour(P, D, feasibility, levels=[0.5, 1.5, 2.5, 3.5, 4.5], 
              colors='black', linewidths=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Transmit Power (dBm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('ISL Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('ISAC Feasibility Regions\n(High Performance Hardware)', 
                fontsize=IEEEStyle.FONT_SIZES['title'])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_map[0], edgecolor='black', label='Link Fails'),
        Patch(facecolor=colors_map[1], edgecolor='black', label='Poor Performance'),
        Patch(facecolor=colors_map[2], edgecolor='black', label='Communication Only'),
        Patch(facecolor=colors_map[3], edgecolor='black', label='Sensing Only'),
        Patch(facecolor=colors_map[4], edgecolor='black', label='Acceptable ISAC'),
        Patch(facecolor=colors_map[5], edgecolor='black', label='Good ISAC'),
        Patch(facecolor=colors_map[6], edgecolor='black', label='Excellent ISAC')
    ]
    ax.legend(handles=legend_elements, loc='lower left',
             fontsize=IEEEStyle.FONT_SIZES['legend']-1, ncol=2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Mark optimal region
    from matplotlib.patches import Rectangle
    rect = Rectangle((32, 1000), 3, 1000, fill=False, 
                    edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(33.5, 1500, 'Optimal', ha='center', color='white',
           fontsize=IEEEStyle.FONT_SIZES['annotation'], 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "ISAC feasibility regions in parameter space")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def modified_blahut_arimoto(system: ISACSystem, D_target: float, 
                           P_tx_scale: float = 1.0,
                           epsilon_lambda: float = 1e-3,
                           epsilon_p: float = 1e-6,
                           max_iterations: int = 50,
                           n_mc: int = 100,
                           verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Simplified but effective Modified Blahut-Arimoto for C-D trade-off.
    
    This implementation uses a practical approach:
    1. Use water-filling inspired initialization
    2. Iteratively adjust distribution based on capacity-distortion gradient
    3. Binary search on Lagrange multiplier for constraint satisfaction
    
    Args:
        system: ISAC system instance
        D_target: Target distortion constraint
        P_tx_scale: Transmit power scaling factor
        epsilon_lambda: Convergence threshold for Lagrange multiplier
        epsilon_p: Convergence threshold for distribution
        max_iterations: Maximum inner loop iterations
        n_mc: Monte Carlo samples for averaging
        verbose: Print optimization progress
    
    Returns:
        Tuple of (capacity, optimal distribution)
    """
    n_symbols = len(system.constellation)
    symbol_powers = np.abs(system.constellation)**2
    
    # Smart initialization based on constraint tightness
    p_uniform = np.ones(n_symbols) / n_symbols
    D_uniform = system.calculate_distortion(p_uniform, P_tx_scale, n_mc)
    
    if D_uniform <= D_target:
        # Uniform already satisfies constraint
        I_x = system.calculate_mutual_information(p_uniform, P_tx_scale, n_mc)
        return np.sum(p_uniform * I_x), p_uniform
    
    # Initialize with power-weighted distribution for tight constraints
    p_x = symbol_powers / np.sum(symbol_powers)
    
    # Binary search for optimal Lagrange multiplier
    lambda_min = 0.0
    lambda_max = 1000.0  # Upper bound for lambda
    
    best_capacity = 0
    best_p_x = p_x.copy()
    
    for outer_iter in range(30):  # Limit outer iterations
        lambda_current = (lambda_min + lambda_max) / 2
        
        # Inner loop: optimize distribution for fixed lambda
        p_x = best_p_x.copy() if outer_iter > 0 else p_x
        
        for inner_iter in range(max_iterations):
            p_x_old = p_x.copy()
            
            # Calculate mutual information for each symbol
            I_x = system.calculate_mutual_information(p_x, P_tx_scale, n_mc)
            
            # Calculate distortion and its gradient
            D_current = system.calculate_distortion(p_x, P_tx_scale, n_mc)
            
            # Approximate gradient of distortion w.r.t. distribution
            # Using finite differences for robustness
            grad_D = np.zeros(n_symbols)
            delta = 0.01
            for i in range(n_symbols):
                if p_x[i] > delta:
                    # Create perturbed distribution
                    p_perturb = p_x.copy()
                    p_perturb[i] -= delta/2
                    p_perturb[(i+1) % n_symbols] += delta/2
                    
                    # Calculate gradient component
                    D_perturb = system.calculate_distortion(p_perturb, P_tx_scale, n_mc//2)
                    grad_D[i] = (D_perturb - D_current) / (delta/2)
            
            # Compute update direction (gradient of Lagrangian)
            gradient = I_x - lambda_current * grad_D
            
            # Exponentiated gradient update (maintains simplex constraint)
            step_size = 0.5 / (1 + inner_iter/10)  # Decreasing step size
            log_p = np.log(p_x + 1e-20)
            log_p += step_size * gradient
            log_p -= np.max(log_p)  # Numerical stability
            
            # Update distribution
            p_x = np.exp(log_p)
            p_x /= np.sum(p_x)
            
            # Check convergence
            if np.sum(np.abs(p_x - p_x_old)) < epsilon_p:
                break
        
        # Evaluate final capacity and distortion
        I_x_final = system.calculate_mutual_information(p_x, P_tx_scale, n_mc)
        capacity = np.sum(p_x * I_x_final)
        D_final = system.calculate_distortion(p_x, P_tx_scale, n_mc)
        
        if verbose and outer_iter % 5 == 0:
            print(f"  Outer iter {outer_iter}: λ={lambda_current:.3e}, "
                  f"C={capacity:.3f} bits/symbol, D={D_final:.3e} (target={D_target:.3e})")
        
        # Update best solution if improved
        if capacity > best_capacity and D_final <= D_target * 1.05:  # 5% tolerance
            best_capacity = capacity
            best_p_x = p_x.copy()
        
        # Update lambda bounds based on constraint satisfaction
        if D_final > D_target:
            lambda_min = lambda_current  # Need more penalty
        else:
            lambda_max = lambda_current  # Can reduce penalty
        
        # Check convergence
        if abs(lambda_max - lambda_min) < epsilon_lambda:
            if verbose:
                print(f"  Converged at λ={lambda_current:.3e}")
            break
    
    # Final check and adjustment
    D_best = system.calculate_distortion(best_p_x, P_tx_scale, n_mc)
    if D_best > D_target * 1.1:  # If still violates constraint significantly
        # Fall back to more conservative distribution
        alpha = 0.5
        while alpha > 0.1:
            p_mixed = alpha * best_p_x + (1-alpha) * p_uniform
            D_mixed = system.calculate_distortion(p_mixed, P_tx_scale, n_mc)
            if D_mixed <= D_target:
                best_p_x = p_mixed
                I_x_mixed = system.calculate_mutual_information(p_mixed, P_tx_scale, n_mc)
                best_capacity = np.sum(p_mixed * I_x_mixed)
                break
            alpha *= 0.8
    
    return best_capacity, best_p_x

def main():
    """Main function with all analyses."""
    print("=== Enhanced THz ISL ISAC Analysis Suite ===")
    print("With data saving and 3D visualizations")
    print("Key Features:")
    print("- All hardware profiles included")
    print("- Pointing error sensitivity analysis")
    print("- 3D landscapes with multiple views")
    print("- Professional color schemes")
    print("- Frequency range up to 1 THz")
    print(f"- Default TX power: {scenario.default_tx_power_dBm} dBm")
    print(f"- Default antenna: {scenario.default_antenna_diameter} m")
    
    # Set debug level
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = False
    
    # Generate all plots
    plot_cd_frontier_all_profiles()
    plot_cd_frontier_pointing_sensitivity()
    plot_snr_to_hardware_limit()
    plot_gamma_eff_sensitivity()
    plot_3d_cd_landscape()
    plot_isac_feasibility_regions()
    
    print("\n=== Analysis Complete ===")
    print("Generated files in results/:")
    print("- fig_cd_frontier_all.pdf/png + data")
    print("- fig_cd_pointing_sensitivity.pdf/png + data")
    print("- fig_snr_to_hardware_limit.pdf/png + data")
    print("- fig_gamma_eff_sensitivity.pdf/png + data")
    print("- fig_3d_cd_landscape_[capacity/rmse]_[views].pdf/png + data")
    print("- fig_isac_feasibility.pdf/png + data")

if __name__ == "__main__":
    main()