#!/usr/bin/env python3
"""
capacity_simulation.py - IEEE Publication Style with Individual Plots
Updated with data saving, 3D plots, 1THz support, and integrated diagnostics
"""
from diagnostics import diagnostics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, Dict, List
from scipy.linalg import inv
from tqdm import tqdm
import os
from scipy.special import erf, erfc

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

class EnhancedISACSystem:
    """Enhanced THz ISL ISAC system with IEEE publication style and pointing error."""
    
    def __init__(self, hardware_profile: str, f_c: float = 300e9, 
             distance: float = 2000e3, n_pilots: int = 64,
             antenna_diameter: float = None,
             tx_power_dBm: float = None):
        """Initialize with enhanced parameters."""
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        self.antenna_diameter = antenna_diameter or scenario.default_antenna_diameter
        self.tx_power_dBm = tx_power_dBm or scenario.default_tx_power_dBm
        
        # Calculate system parameters
        self.lambda_c = PhysicalConstants.c / f_c
        self._calculate_link_budget()
        
        # PA parameters
        self.bussgang_gain = self._calculate_bussgang_gain()
        
        # Constellation
        self.constellation = self._create_constellation()
        
        # Add diagnostics for first initialization only
        if not hasattr(self.__class__, '_diagnostics_logged'):
            self._log_diagnostics()
            self.__class__._diagnostics_logged = True

    def _log_diagnostics(self):
        """Log key system parameters for diagnostics."""
        # Physical parameters
        diagnostics.add_key_metric("Physical", "Frequency_GHz", self.f_c/1e9, 
                                (10, 1000), "GHz")
        diagnostics.add_key_metric("Physical", "Distance_km", self.distance/1e3,
                                (100, 10000), "km")
        diagnostics.add_key_metric("Physical", "Antenna_diameter_m", self.antenna_diameter,
                                (0.1, 5), "m")
        
        # Link budget
        diagnostics.add_key_metric("LinkBudget", "TxPower_dBm", self.tx_power_dBm,
                                (0, 40), "dBm")
        diagnostics.add_key_metric("LinkBudget", "PathLoss_dB", self.path_loss_dB,
                                (150, 250), "dB")
        diagnostics.add_key_metric("LinkBudget", "RxPower_dBm", self.P_rx_dBm,
                                (-80, 0), "dBm")
        diagnostics.add_key_metric("LinkBudget", "LinkMargin_dB", self.link_margin_dB,
                                (0, 50), "dB")
        
        # Hardware impairments
        diagnostics.add_key_metric("Hardware", f"Gamma_eff_{self.profile.name}", 
                                self.profile.Gamma_eff, (1e-4, 1e-1), "")
        diagnostics.add_key_metric("Hardware", f"PhaseNoiseVar_{self.profile.name}_rad2", 
                                self.profile.phase_noise_variance, (0, 10), "rad²")
        diagnostics.add_key_metric("Hardware", f"BussgangGain_{self.profile.name}", 
                                np.abs(self.bussgang_gain), (0.5, 1.0), "")
        
        # Channel
        diagnostics.add_key_metric("Channel", "ChannelGain_linear", 
                                np.abs(self.channel_gain), (0, 1), "")
    
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
        
        # Noise parameters - RENAMED FOR CLARITY
        self.noise_figure_dB = 8
        self.bandwidth_Hz = self.profile.signal_bandwidth_Hz
        
        # Calculate total noise power (not PSD)
        noise_temp_K = 290 * 10**(self.noise_figure_dB/10)
        self.noise_power_watts = PhysicalConstants.k * noise_temp_K * self.bandwidth_Hz  # Renamed from N_0
        self.noise_power_dBm = 10 * np.log10(self.noise_power_watts * 1000)
        
        # Channel gain calculation
        G_tx_linear = 10**(self.G_tx_dB/10)
        G_rx_linear = 10**(self.G_rx_dB/10)
        path_loss_linear = 10**(-self.path_loss_dB/10)
        
        # Total channel gain magnitude |g|
        self.channel_gain = np.sqrt(G_tx_linear * G_rx_linear * path_loss_linear)
        
        # Link margin calculation
        self.link_margin_dB = self.P_rx_dBm - self.noise_power_dBm
    
    def _calculate_bussgang_gain(self, input_backoff_dB: float = 7.0) -> float:
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
    
    # 替换 calculate_sinr_mc 方法
    def calculate_sinr_mc(self, symbol: complex, avg_power: float, P_tx_scale: float,
                    n_mc: int = 100) -> float:
        """Calculate SINR for given symbol with Monte Carlo pointing error averaging."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        symbol_power = np.abs(symbol)**2
        
        # Monte Carlo averaging over pointing error
        pointing_losses = scenario.sample_pointing_loss(
            self.f_c, self.antenna_diameter, n_samples=n_mc
        )
        
        # Ensure real values by taking absolute value of channel gain squared
        P_rx_signal_base = P_tx * symbol_power * np.abs(self.channel_gain)**2 * np.abs(self.bussgang_gain)**2
        P_rx_signal_avg = P_rx_signal_base * np.mean(pointing_losses)
        
        N_thermal = self.noise_power_watts  # Changed from self.N_0
        N_hw = P_rx_signal_avg * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        
        N_total = N_thermal + N_hw * phase_penalty
        sinr = P_rx_signal_avg / N_total
        
        # Ensure real and positive
        return np.real(np.abs(sinr))
    
    def calculate_mutual_information(self, p_x: np.ndarray, P_tx_scale: float = 1.0,
                                n_mc: int = 100) -> np.ndarray:
        """Calculate mutual information for each symbol with MC averaging."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr_mc(symbol, avg_power, P_tx_scale, n_mc)
            # Ensure real value for log2
            I_x[i] = np.real(np.log2(1 + np.abs(sinr)))
            
        return I_x
    
    # 替换 calculate_capacity_vs_snr 方法
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
    
    # 替换 calculate_bfim_observable_mc 方法
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
# INDIVIDUAL PLOT FUNCTIONS WITH INTEGRATED DIAGNOSTICS
# =========================================================================

# 添加辅助函数来生成完整前沿
def generate_cd_frontier(system, P_tx_scales=None, n_mc=50):
    """Generate C-D frontier by power scaling only (simpler version)."""
    
    # Power scaling range
    P_tx_scales = np.logspace(-0.5, +0.5, n_points)  # ±3dB range
    
    # Keep uniform distribution
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    
    all_D = []
    all_C = []
    
    for P_scale in tqdm(P_tx_scales, desc="    Power sweep", leave=False):
        try:
            # Calculate capacity
            I_x = system.calculate_mutual_information(p_uniform, P_tx_scale=P_scale, n_mc=50)
            capacity = np.mean(I_x)
            
            # Calculate distortion
            distortion = system.calculate_distortion(p_uniform, P_tx_scale=P_scale, n_mc=50)
            
            if 0 < distortion < 1e10 and capacity > 0:
                all_D.append(distortion)
                all_C.append(capacity)
        except:
            continue
    
    if len(all_D) == 0:
        return np.array([]), np.array([])
    
    # Sort and extract Pareto frontier
    sorted_idx = np.argsort(all_D)
    D_sorted = np.array(all_D)[sorted_idx]
    C_sorted = np.array(all_C)[sorted_idx]
    
    # Extract Pareto optimal points
    pareto_D = []
    pareto_C = []
    max_C = -np.inf
    
    for i in range(len(D_sorted)):
        if C_sorted[i] > max_C:
            pareto_D.append(D_sorted[i])
            pareto_C.append(C_sorted[i])
            max_C = C_sorted[i]
    
    return np.array(pareto_D), np.array(pareto_C)



# 文件: capacity_simulation.py
# 在plot_cd_frontier函数前添加辅助函数

def generate_cd_frontier_with_overhead(system, K_total=1024, n_points=50):
    """Generate C-D frontier with pilot overhead penalty."""
    
    # Power and pilot configurations
    P_tx_scales = np.logspace(-2, +1, n_points)
    pilot_counts = [8, 16, 32, 64, 128, 256]
    pilot_counts = [M for M in pilot_counts if M < K_total]
    
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    
    all_points = []
    original_pilots = system.n_pilots
    
    for M in pilot_counts:
        system.n_pilots = M
        alpha = M / K_total  # Pilot overhead
        
        for P_scale in P_tx_scales:
            try:
                # Per-symbol capacity
                I_x = system.calculate_mutual_information(p_uniform, P_tx_scale=P_scale, n_mc=50)
                C_per = np.mean(I_x)
                
                # Effective capacity with overhead
                C_eff = (1.0 - alpha) * C_per
                
                # Distortion
                distortion = system.calculate_distortion(p_uniform, P_tx_scale=P_scale, n_mc=50)
                
                if 0 < distortion < 1e10 and C_eff > 0:
                    all_points.append((distortion, C_eff))
            except:
                continue
    
    system.n_pilots = original_pilots
    
    if len(all_points) == 0:
        return np.array([]), np.array([])
    
    # Extract Pareto frontier
    all_points.sort(key=lambda x: x[0])
    pareto_D = []
    pareto_C = []
    max_C = -np.inf
    
    for D, C in all_points:
        if C > max_C:
            pareto_D.append(D)
            pareto_C.append(C)
            max_C = C
    
    return np.array(pareto_D), np.array(pareto_C)

def plot_cd_frontier(save_name='fig_cd_frontier'):
    """Plot C-D frontier for all hardware profiles with complete curves."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    data_to_save = {
        'hardware_profiles': profiles_to_plot
    }
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        print(f"  Processing {profile_name}...")
        system = EnhancedISACSystem(profile_name)
        
        # Generate complete C-D frontier
        distortions, capacities = generate_cd_frontier(
            system,
            P_tx_scales=np.logspace(-2, +2, 100),
            n_mc=50
        )
        
        if distortions.size > 0:
            ranging_rmse_mm = np.sqrt(distortions) * 1000
            
            data_to_save[f'ranging_rmse_mm_{profile_name}'] = ranging_rmse_mm.tolist()
            data_to_save[f'capacity_{profile_name}'] = capacities.tolist()
            data_to_save[f'num_points_{profile_name}'] = len(capacities)
            
            # Add to diagnostics
            diagnostics.add_key_metric(
                "CD_Frontier", 
                f"{profile_name}_MaxCapacity", 
                np.max(capacities), 
                (0, 10), 
                "bits/symbol"
            )
            diagnostics.add_key_metric(
                "CD_Frontier",
                f"{profile_name}_MinRMSE",
                np.min(ranging_rmse_mm),
                (0.001, 1000),
                "mm"
            )
            
            # Plot complete frontier curve
            ax.plot(ranging_rmse_mm, capacities,
                   color=colors[idx], 
                   linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                   linestyle='-',
                   label=f'{profile_name.replace("_", " ")}')
            
            # Add markers at selected points
            marker_indices = np.linspace(0, len(ranging_rmse_mm)-1, 
                                        min(10, len(ranging_rmse_mm)), dtype=int)
            ax.plot(ranging_rmse_mm[marker_indices], capacities[marker_indices],
                   color=colors[idx],
                   linestyle='None',
                   marker=markers[idx], 
                   markersize=IEEEStyle.LINE_PROPS['markersize'],
                   markerfacecolor='white', 
                   markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'])
    
    # Adjust x-axis range
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 100)
    ax.set_ylim(0, 10)
    
    # Add feasibility regions
    ax.axhspan(2.0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.axvspan(ax.get_xlim()[0], 1.0, alpha=0.1, color='blue')
    
    # Add performance thresholds
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(1e-2, 2.1, 'Good communication', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='green')
    
    ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.8, ax.get_ylim()[1]*0.7, 'Sub-mm\nsensing', ha='right',
           fontsize=IEEEStyle.FONT_SIZES['annotation'], color='blue')
    
    # Labels
    ax.set_xlabel('Ranging RMSE (mm)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Capacity-Distortion Trade-off (Power Scaling)', 
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='upper right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Capacity-Distortion frontier for all hardware profiles")
    
    print(f"Saved: results/{save_name}.pdf/png and data")
    for profile in profiles_to_plot:
        if f'num_points_{profile}' in data_to_save:
            print(f"  {profile}: {data_to_save[f'num_points_{profile}']} frontier points")


def plot_capacity_vs_snr(save_name='fig_capacity_vs_snr'):
    """Plot capacity vs SNR for all hardware profiles - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    profiles = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    snr_dB_array = np.linspace(-10, 60, 71)
    
    # Data storage
    data_to_save = {
        'snr_dB': snr_dB_array.tolist(),
        'hardware_profiles': profiles
    }
    
    for idx, profile_name in enumerate(profiles):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        print(f"  Processing {profile_name}...")
        system = EnhancedISACSystem(profile_name)
        results = system.calculate_capacity_vs_snr(snr_dB_array, n_mc=100)
        
        data_to_save[f'capacity_{profile_name}'] = results['capacity'].tolist()
        data_to_save[f'ceiling_{profile_name}'] = results['ceiling']
        
        # Add to diagnostics
        diagnostics.add_key_metric(
            "Capacity_vs_SNR",
            f"{profile_name}_Ceiling",
            results['ceiling'],
            (0, 20),
            "bits/symbol"
        )
        diagnostics.add_key_metric(
            "Capacity_vs_SNR",
            f"{profile_name}_MaxCapacity",
            np.max(results['capacity']),
            (0, 20),
            "bits/symbol"
        )
        
        # Find SNR where capacity reaches 95% of ceiling
         # 修正这一行：添加 phase_noise_variance 参数
        hw_limit_snr = DerivedParameters.find_snr_for_hardware_limit(
            system.profile.Gamma_eff, 
            system.profile.phase_noise_variance,  # 添加这个参数
            0.95
        )
        diagnostics.add_key_metric(
            "Capacity_vs_SNR",
            f"{profile_name}_SNR_95pct_ceiling",
            hw_limit_snr,
            (20, 60),
            "dB"
        )
        
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
        ax.axvline(x=hw_limit_snr, color=colors[idx], 
                  linestyle=':', alpha=0.3, linewidth=1.2)
    
    # Add regions
    ax.axvspan(-10, 10, alpha=0.1, color='blue')
    ax.axvspan(40, 60, alpha=0.1, color='red')
    ax.text(0, 0.5, 'Power\nLimited', ha='center', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'])
    ax.text(50, 0.5, 'Hardware\nLimited', ha='center', 
           fontsize=IEEEStyle.FONT_SIZES['annotation'])
    
    # Labels
    ax.set_xlabel('SNR (dB)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Capacity vs. SNR with Hardware Limitations',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='lower right', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(0, 10)
    ax.set_xlim(-10, 60)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Capacity vs SNR for all hardware profiles")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_capacity_vs_frequency(save_name='fig_capacity_vs_frequency'):
    """Plot capacity vs frequency at different distances."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    # Parameters
    frequencies_GHz = simulation.frequency_sweep_GHz
    distances_km = [500, 1000, 2000, 5000]
    hardware_profile = "High_Performance"
    
    # Data storage
    data_to_save = {
        'frequency_GHz': frequencies_GHz.tolist(),
        'distances_km': distances_km,
        'hardware_profile': hardware_profile
    }
    
    for idx, d_km in enumerate(distances_km):
        capacities = []
        print(f"  Processing distance {d_km} km...")
        
        for f_GHz in tqdm(frequencies_GHz, desc=f"    Frequency sweep", leave=False):
            system = EnhancedISACSystem(
                hardware_profile, f_c=f_GHz*1e9, distance=d_km*1e3
            )
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacities.append(np.mean(I_x))
        
        data_to_save[f'capacity_{d_km}km'] = capacities
        
        # Add key metrics to diagnostics
        diagnostics.add_key_metric(
            "Frequency_Analysis",
            f"Capacity_at_1THz_{d_km}km",
            capacities[-1] if f_GHz == 1000 else 0,
            (0, 10),
            "bits/symbol"
        )
        
        ax.plot(frequencies_GHz, capacities, 
                color=colors[idx], linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                marker=markers[idx], markersize=IEEEStyle.LINE_PROPS['markersize'],
                markevery=2, markerfacecolor='white',
                markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                label=f'{d_km} km')
    
    # Highlight 1THz
    ax.axvline(x=1000, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(1000, ax.get_ylim()[1]*0.95, '1 THz', 
           ha='center', fontsize=IEEEStyle.FONT_SIZES['annotation'], color='red')
    
    ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Capacity vs. Frequency at Different Distances', 
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.1)
    ax.set_xlim(50, 1100)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Capacity vs frequency at different distances")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_capacity_vs_distance(save_name='fig_capacity_vs_distance'):
    """Plot capacity vs distance at different frequencies."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    # Parameters
    distances_km = np.linspace(500, 5000, 20)
    frequencies_GHz = [100, 300, 600, 1000]
    hardware_profile = "High_Performance"
    
    # Data storage
    data_to_save = {
        'distance_km': distances_km.tolist(),
        'frequencies_GHz': frequencies_GHz,
        'hardware_profile': hardware_profile
    }
    
    for idx, f_GHz in enumerate(frequencies_GHz):
        capacities = []
        print(f"  Processing frequency {f_GHz} GHz...")
        
        for d_km in tqdm(distances_km, desc=f"    Distance sweep", leave=False):
            system = EnhancedISACSystem(
                hardware_profile, f_c=f_GHz*1e9, distance=d_km*1e3
            )
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacities.append(np.mean(I_x))
        
        data_to_save[f'capacity_{f_GHz}GHz'] = capacities
        
        # Add key metrics
        diagnostics.add_key_metric(
            "Distance_Analysis",
            f"Capacity_{f_GHz}GHz_at_2000km",
            capacities[np.argmin(np.abs(distances_km - 2000))],
            (0, 10),
            "bits/symbol"
        )
        
        ax.plot(distances_km, capacities, 
                color=colors[idx], linewidth=IEEEStyle.LINE_PROPS['linewidth'],
                marker=markers[idx], markersize=IEEEStyle.LINE_PROPS['markersize'],
                markevery=4, markerfacecolor='white',
                markeredgewidth=IEEEStyle.LINE_PROPS['markeredgewidth'],
                label=f'{f_GHz} GHz')
    
    ax.set_xlabel('Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Capacity vs. Distance at Different Frequencies', 
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.1)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Capacity vs distance at different frequencies")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_hardware_quality_impact(save_name='fig_hardware_quality_impact'):
    """Plot capacity vs hardware quality factor - IEEE style."""
    print(f"\n=== Generating {save_name} ===")
    
    fig, ax = plt.subplots(figsize=IEEEStyle.FIG_SIZES['single'])
    
    gamma_eff_range = np.logspace(-3, -1, 30)
    snr_levels_dB = [20, 30, 40, 50]
    
    # Data storage
    data_to_save = {
        'gamma_eff': gamma_eff_range.tolist(),
        'snr_levels_dB': snr_levels_dB
    }
    
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
        
        data_to_save[f'capacity_snr{snr_dB}dB'] = capacities
        
        # Add diagnostics for key SNR level (40 dB)
        if snr_dB == 40:
            diagnostics.add_key_metric(
                "Hardware_Quality",
                "Capacity_at_Gamma_0.01",
                capacities[np.argmin(np.abs(gamma_eff_range - 0.01))],
                (0, 10),
                "bits/symbol"
            )
        
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
            ax.text(profile.Gamma_eff*1.1, ax.get_ylim()[1]*0.95, 
                   name.split('_')[0], rotation=90, 
                   fontsize=IEEEStyle.FONT_SIZES['annotation']-1, alpha=0.7, va='top')
    
    # Labels
    ax.set_xlabel('Hardware Quality Factor $\Gamma_{eff}$',
                 fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_ylabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
    ax.set_title('Impact of Hardware Quality on Capacity',
                fontsize=IEEEStyle.FONT_SIZES['title'])
    ax.grid(True, **IEEEStyle.GRID_PROPS)
    ax.legend(loc='lower left', fontsize=IEEEStyle.FONT_SIZES['legend'])
    ax.set_ylim(0, 10)
    ax.set_xlim(1e-3, 1e-1)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/{save_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "Hardware quality impact on capacity")
    
    print(f"Saved: results/{save_name}.pdf/png and data")

def plot_3d_capacity_landscape(save_name='fig_3d_capacity_landscape'):
    """Plot 3D capacity landscape with multiple viewing angles."""
    print(f"\n=== Generating {save_name} ===")
    
    # Parameter ranges
    frequencies_GHz = np.linspace(100, 1000, 15)
    distances_km = np.linspace(500, 5000, 15)
    
    # Create meshgrid
    F, D = np.meshgrid(frequencies_GHz, distances_km)
    
    # Fixed parameters
    hardware_profile = "High_Performance"
    
    # Calculate capacity for each point
    capacity_grid = np.zeros_like(F)
    
    print("  Computing 3D capacity landscape...")
    for i in tqdm(range(F.shape[0]), desc="    Distance levels"):
        for j in range(F.shape[1]):
            f_Hz = F[i, j] * 1e9
            d_m = D[i, j] * 1e3
            
            # Create system
            system = EnhancedISACSystem(hardware_profile, f_c=f_Hz, distance=d_m)
            
            # Calculate capacity
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x, n_mc=50)
            capacity_grid[i, j] = np.mean(I_x)
    
    # Add to diagnostics
    diagnostics.add_key_metric(
        "3D_Landscape",
        "Max_Capacity",
        np.max(capacity_grid),
        (0, 10),
        "bits/symbol"
    )
    diagnostics.add_key_metric(
        "3D_Landscape",
        "Min_Capacity",
        np.min(capacity_grid),
        (0, 10),
        "bits/symbol"
    )
    diagnostics.add_key_metric(
        "3D_Landscape",
        "Capacity_1THz_2000km",
        capacity_grid[np.argmin(np.abs(distances_km - 2000)), 
                     np.argmin(np.abs(frequencies_GHz - 1000))],
        (0, 10),
        "bits/symbol"
    )
    
    # Data storage
    data_to_save = {
        'frequency_GHz': frequencies_GHz.tolist(),
        'distance_km': distances_km.tolist(),
        'capacity_grid': capacity_grid.tolist(),
        'hardware_profile': hardware_profile
    }
    
    # Create multiple views
    viewing_angles = [
        (25, 45, 'default'),
        (15, 60, 'frequency_emphasis'),
        (30, 15, 'distance_emphasis'),
        (45, 45, 'isometric')
    ]
    
    for elev, azim, view_name in viewing_angles:
        fig = plt.figure(figsize=IEEEStyle.FIG_SIZES['3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(F, D, capacity_grid, cmap='viridis', 
                              edgecolor='none', alpha=0.8)
        
        # Add contour lines at the bottom
        contours = ax.contour(F, D, capacity_grid, zdir='z', offset=0, 
                              cmap='viridis', alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Frequency (GHz)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_ylabel('Distance (km)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_zlabel('Capacity (bits/symbol)', fontsize=IEEEStyle.FONT_SIZES['label'])
        ax.set_title('THz ISL ISAC Capacity Landscape\n(High Performance Hardware)', 
                    fontsize=IEEEStyle.FONT_SIZES['title'], pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Highlight 1THz
        ax.plot([1000, 1000], [distances_km[0], distances_km[-1]], 
               [0, 0], 'r--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_name}_{view_name}.pdf', 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/{save_name}_{view_name}.png', 
                   format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    data_saver.save_data(save_name, data_to_save,
                       "3D capacity landscape data")
    
    print(f"Saved: results/{save_name}_[views].pdf/png and data")

def simple_cd_optimization(system: EnhancedISACSystem, D_target: float,
                          max_iterations: int = 20, n_mc: int = 100) -> Tuple[float, np.ndarray]:
    """Simplified C-D optimization for capacity plots.
    
    Uses a gradient-free approach suitable for visualization purposes.
    """
    n_symbols = len(system.constellation)
    symbol_powers = np.abs(system.constellation)**2
    
    # Check if uniform distribution meets constraint
    p_uniform = np.ones(n_symbols) / n_symbols
    D_uniform = system.calculate_distortion(p_uniform, n_mc=n_mc)
    
    if D_uniform <= D_target:
        I_x = system.calculate_mutual_information(p_uniform, n_mc=n_mc)
        return np.sum(p_uniform * I_x), p_uniform
    
    # Try different power allocations
    best_capacity = 0
    best_p_x = p_uniform
    
    # Test various distributions between uniform and max-power
    for alpha in np.linspace(0, 1, 20):
        # Interpolate between uniform and power-focused
        p_test = (1-alpha) * p_uniform + alpha * (symbol_powers/np.sum(symbol_powers))
        
        # Check distortion
        D_test = system.calculate_distortion(p_test, n_mc=n_mc)
        
        if D_test <= D_target:
            # Calculate capacity
            I_x = system.calculate_mutual_information(p_test, n_mc=n_mc)
            capacity = np.sum(p_test * I_x)
            
            if capacity > best_capacity:
                best_capacity = capacity
                best_p_x = p_test.copy()
    
    return best_capacity, best_p_x

def main():
    """Main function to generate all capacity analysis plots."""
    print("=== THz ISL ISAC Capacity Analysis (IEEE Style) ===")
    print("With unified diagnostics and data collection")
    
    # Clear previous diagnostics
    diagnostics.results.clear()
    diagnostics.warnings.clear()
    diagnostics.key_metrics.clear()
    
    # Note about observability
    print("\nNOTE: Analysis based on single ISL (2 observable parameters)")
    
    # Run all analyses with diagnostics
    print("\n" + "="*70)
    print("RUNNING ANALYSES WITH DIAGNOSTICS")
    print("="*70)
    
    # Each plot function will add its results to diagnostics
    plot_cd_frontier()
    plot_capacity_vs_snr()
    plot_capacity_vs_frequency()
    plot_capacity_vs_distance()
    plot_hardware_quality_impact()
    plot_3d_capacity_landscape()
    
    # Print diagnostics summary
    diagnostics.print_summary()
    
    # Save comprehensive report
    diagnostics.save_comprehensive_report("capacity_analysis_complete")
    
    print("\n=== Capacity Analysis Complete ===")

if __name__ == "__main__":
    main()