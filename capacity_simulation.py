#!/usr/bin/env python3
"""
capacity_simulation.py - Enhanced version with comprehensive C-D analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
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
    ObservableParameters
)

# Set publication-quality plot defaults
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
})

colors = sns.color_palette("husl", 6)

class EnhancedISACSystem:
    """Enhanced THz ISL ISAC system with comprehensive analysis capabilities."""
    
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
        
        # Print warning about observability
        if not hasattr(self.__class__, '_observability_warned'):
            ObservableParameters.print_observability_warning()
            self.__class__._observability_warned = True
        
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
        elif modulation == '16QAM':
            # 16-QAM for higher data rates
            real_parts = np.array([-3, -1, 1, 3])
            imag_parts = np.array([-3, -1, 1, 3])
            constellation = []
            for r in real_parts:
                for i in imag_parts:
                    constellation.append(r + 1j*i)
            constellation = np.array(constellation)
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
    
    def calculate_capacity_vs_snr(self, snr_dB_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate capacity vs SNR showing hardware ceiling."""
        capacities = []
        sinr_effective = []
        
        for snr_dB in snr_dB_array:
            # Override noise floor to achieve target SNR
            snr_linear = 10**(snr_dB/10)
            
            # Uniform distribution
            p_x = np.ones(len(self.constellation)) / len(self.constellation)
            avg_power = 1.0
            
            # Calculate effective SINR with hardware impairments
            P_signal = self.P_tx_watts * (self.channel_gain**2) * (self.bussgang_gain**2)
            N_0_target = P_signal / snr_linear
            
            # Hardware noise
            N_hw = P_signal * self.profile.Gamma_eff * np.exp(self.profile.phase_noise_variance)
            
            # Total noise
            N_total = N_0_target + N_hw
            
            # Effective SINR
            sinr_eff = P_signal / N_total
            sinr_effective.append(10 * np.log10(sinr_eff))
            
            # Capacity
            capacity = np.log2(1 + sinr_eff)
            capacities.append(capacity)
        
        # Calculate hardware ceiling
        ceiling = DerivedParameters.capacity_ceiling(
            self.profile.Gamma_eff, self.profile.phase_noise_variance
        )
        
        return {
            'snr_dB': snr_dB_array,
            'capacity': np.array(capacities),
            'sinr_effective': np.array(sinr_effective),
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
            # Only trace of observable parameters
            distortion = J_B_inv[0,0]  # Range variance only
            
            if distortion < 0 or distortion > 1e15:
                return 1e10
                
        except np.linalg.LinAlgError:
            return 1e10
            
        return distortion

def plot_comprehensive_cd_analysis():
    """Generate comprehensive C-D analysis with all requested features."""
    print("\n=== Generating Comprehensive C-D Analysis ===")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Subplot 1: C-D frontier for different hardware
    ax1 = fig.add_subplot(gs[0, :2])
    plot_cd_frontier_comparison(ax1)
    
    # Subplot 2: SNR regions
    ax2 = fig.add_subplot(gs[0, 2])
    plot_snr_regions(ax2)
    
    # Subplot 3: Hardware parameter sweep at fixed SNR
    ax3 = fig.add_subplot(gs[1, :])
    plot_hardware_sweep_fixed_snr(ax3)
    
    # Subplot 4: SNR sweep with good hardware
    ax4 = fig.add_subplot(gs[2, :])
    plot_snr_sweep_good_hardware(ax4)
    
    plt.suptitle('Comprehensive THz ISL ISAC Analysis', fontsize=18)
    plt.tight_layout()
    plt.savefig('comprehensive_cd_analysis.pdf', format='pdf', dpi=300)
    plt.savefig('comprehensive_cd_analysis.png', format='png', dpi=300)
    plt.show()

def plot_cd_frontier_comparison(ax):
    """Plot C-D frontier for different hardware profiles."""
    profiles_to_plot = ["State_of_Art", "High_Performance", "SWaP_Efficient", "Low_Cost"]
    
    for idx, profile_name in enumerate(profiles_to_plot):
        if profile_name not in HARDWARE_PROFILES:
            continue
            
        system = EnhancedISACSystem(profile_name, antenna_diameter=1.0, tx_power_dBm=30)
        
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
            ax.plot(ranging_rmse_mm, capacities,
                   color=colors[idx], linewidth=2.5,
                   marker=['o', 's', '^', 'D'][idx], markersize=7,
                   label=f'{profile_name.replace("_", " ")} (Γ={HARDWARE_PROFILES[profile_name].Gamma_eff})')
    
    # Add feasibility regions
    ax.axhspan(2.0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.axvspan(ax.get_xlim()[0], 1.0, alpha=0.1, color='blue')
    
    ax.set_xlabel('Ranging RMSE [mm]', fontsize=12)
    ax.set_ylabel('Capacity [bits/symbol]', fontsize=12)
    ax.set_title('C-D Trade-off for Different Hardware Profiles', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(left=0.01)

def plot_snr_regions(ax):
    """Plot SNR operational regions."""
    snr_dB = np.linspace(-10, 50, 100)
    
    # Define regions
    power_limited = snr_dB < 10
    transition = (snr_dB >= 10) & (snr_dB < 30)
    hardware_limited = snr_dB >= 30
    
    # Create stacked area plot
    ax.fill_between(snr_dB[power_limited], 0, 1, alpha=0.7, color='blue', 
                   label='Power-Limited')
    ax.fill_between(snr_dB[transition], 0, 1, alpha=0.7, color='yellow',
                   label='Transition')
    ax.fill_between(snr_dB[hardware_limited], 0, 1, alpha=0.7, color='red',
                   label='Hardware-Limited')
    
    ax.set_xlabel('SNR [dB]', fontsize=12)
    ax.set_ylabel('Region', fontsize=12)
    ax.set_title('Operational Regions', fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xlim(-10, 50)
    ax.legend(loc='center')
    ax.set_yticks([])

def plot_hardware_sweep_fixed_snr(ax):
    """Plot performance vs hardware quality at fixed SNR."""
    gamma_eff_range = np.logspace(-3, -1, 30)
    snr_levels_dB = [10, 20, 30, 40]
    
    for snr_dB in snr_levels_dB:
        capacities = []
        
        for gamma_eff in gamma_eff_range:
            # Create custom profile
            custom_profile = HARDWARE_PROFILES["Custom"]
            original_gamma = custom_profile.Gamma_eff
            custom_profile.Gamma_eff = gamma_eff
            
            # Calculate capacity at this SNR
            snr_linear = 10**(snr_dB/10)
            phase_factor = np.exp(-custom_profile.phase_noise_variance)
            sinr_eff = snr_linear / (1 + snr_linear * gamma_eff)
            capacity = np.log2(1 + sinr_eff * phase_factor)
            capacities.append(capacity)
            
            # Restore
            custom_profile.Gamma_eff = original_gamma
        
        ax.semilogx(gamma_eff_range, capacities, linewidth=2.5,
                   label=f'SNR = {snr_dB} dB')
    
    # Mark hardware profiles
    for name, profile in HARDWARE_PROFILES.items():
        if name != "Custom":
            ax.axvline(x=profile.Gamma_eff, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=12)
    ax.set_ylabel('Capacity [bits/symbol]', fontsize=12)
    ax.set_title('Capacity vs Hardware Quality at Fixed SNR', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_snr_sweep_good_hardware(ax):
    """Plot performance vs SNR for good hardware."""
    system_good = EnhancedISACSystem("State_of_Art", antenna_diameter=1.0, tx_power_dBm=30)
    system_typical = EnhancedISACSystem("High_Performance", antenna_diameter=1.0, tx_power_dBm=30)
    
    snr_dB_array = np.linspace(-10, 50, 100)
    
    # Calculate for both systems
    results_good = system_good.calculate_capacity_vs_snr(snr_dB_array)
    results_typical = system_typical.calculate_capacity_vs_snr(snr_dB_array)
    
    # Plot capacity
    ax.plot(snr_dB_array, results_good['capacity'], 'b-', linewidth=3,
           label='State of Art')
    ax.plot(snr_dB_array, results_typical['capacity'], 'r-', linewidth=3,
           label='High Performance')
    
    # Add ceilings
    ax.axhline(y=results_good['ceiling'], color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=results_typical['ceiling'], color='r', linestyle='--', alpha=0.5)
    
    # Add regions
    ax.axvspan(-10, 10, alpha=0.1, color='blue')
    ax.axvspan(30, 50, alpha=0.1, color='red')
    ax.axvspan(10, 30, alpha=0.1, color='yellow')
    
    # Annotations
    ax.text(0, results_good['ceiling']*0.9, 'Power-Limited', 
           ha='center', fontsize=10, style='italic')
    ax.text(40, results_good['ceiling']*0.9, 'Hardware-Limited',
           ha='center', fontsize=10, style='italic')
    
    ax.set_xlabel('SNR [dB]', fontsize=12)
    ax.set_ylabel('Capacity [bits/symbol]', fontsize=12)
    ax.set_title('Capacity vs SNR with Good Hardware', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 8)

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
            # Increase power on stronger symbols
            symbol_powers = np.abs(system.constellation)**2
            gradient = symbol_powers - np.mean(symbol_powers)
        else:
            # Increase power on symbols with higher MI
            gradient = I_x - np.mean(I_x)
        
        # Update distribution
        p_x = p_x * np.exp(step_size * gradient)
        p_x /= np.sum(p_x)
        
        # Check new distortion
        D_current = system.calculate_distortion(p_x)
    
    capacity = np.sum(p_x * I_x)
    return capacity, p_x

def plot_hardware_feasibility_map():
    """Generate feasibility map for different hardware configurations."""
    print("\n=== Generating Hardware Feasibility Map ===")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Parameter ranges
    gamma_eff_range = np.logspace(-3, -1, 20)
    phase_noise_range = np.logspace(-3, -1, 20)
    
    GAMMA, PHASE = np.meshgrid(gamma_eff_range, phase_noise_range)
    
    # Calculate feasibility metric (capacity ceiling)
    feasibility = np.zeros_like(GAMMA)
    
    for i in range(GAMMA.shape[0]):
        for j in range(GAMMA.shape[1]):
            ceiling = DerivedParameters.capacity_ceiling(GAMMA[i,j], PHASE[i,j])
            feasibility[i,j] = ceiling
    
    # Create contour plot
    levels = np.linspace(0, 8, 17)
    cs = ax.contourf(GAMMA, PHASE, feasibility, levels=levels, cmap='viridis')
    
    # Add contour lines
    ax.contour(GAMMA, PHASE, feasibility, levels=[1, 2, 3, 4, 5], 
              colors='white', linewidths=1, alpha=0.5)
    
    # Mark existing hardware profiles
    for name, profile in HARDWARE_PROFILES.items():
        if name != "Custom":
            ax.scatter(profile.Gamma_eff, profile.phase_noise_variance,
                      s=100, marker='*', edgecolors='red', linewidths=2,
                      label=name.replace('_', ' '))
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Capacity Ceiling [bits/symbol]', fontsize=12)
    
    # Formatting
    ax.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=12)
    ax.set_ylabel('Phase Noise Variance σ²_φ [rad²]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Hardware Feasibility Map for THz ISL', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('hardware_feasibility_map.pdf', format='pdf', dpi=300)
    plt.savefig('hardware_feasibility_map.png', format='png', dpi=300)
    plt.show()

def main():
    """Main function with all analyses."""
    print("=== Enhanced THz ISL ISAC Capacity Analysis ===")
    print("\nThis analysis includes:")
    print("1. C-D frontier comparison across hardware profiles")
    print("2. SNR operational regions identification")
    print("3. Hardware parameter sweeps at fixed SNR")
    print("4. SNR sweeps with good hardware")
    print("5. Hardware feasibility mapping\n")
    
    # Note about observability
    print("NOTE: Using single ISL observability model (2 parameters only)")
    print("Full 3D state requires multiple ISLs\n")
    
    # Generate comprehensive analysis
    plot_comprehensive_cd_analysis()
    
    # Generate hardware feasibility map
    plot_hardware_feasibility_map()
    
    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("1. comprehensive_cd_analysis.pdf - All requested analyses in one figure")
    print("2. hardware_feasibility_map.pdf - Shows achievable performance regions")
    
    print("\nKey Findings:")
    print("- Current hardware (Γ_eff ≈ 0.01-0.05) limits capacity to 2-4 bits/symbol")
    print("- State-of-art hardware (Γ_eff ≈ 0.005) could achieve 5+ bits/symbol")
    print("- Hardware limitations dominate above SNR ≈ 20-30 dB")
    print("- Sub-mm ranging requires both good hardware AND sufficient SNR")

if __name__ == "__main__":
    main()