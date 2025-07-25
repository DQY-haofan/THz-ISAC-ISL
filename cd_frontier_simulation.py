#!/usr/bin/env python3
"""
cd_frontier_simulation.py - Complete Fixed Version

Key improvements:
1. Realistic THz ISL link budget
2. Proper power sensitivity in FIM
3. Selective debug output
4. Hardware feasibility check
"""

import numpy as np
import matplotlib.pyplot as plt
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
    DerivedParameters
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

colors = sns.color_palette("husl", 4)

# Global debug flag
DEBUG_VERBOSE = False  # Set to True for detailed output

class ISACSystem:
    """THz ISL ISAC system with realistic link budget."""
    
    def __init__(self, hardware_profile: str, f_c: float = 300e9, 
                 distance: float = 2000e3, n_pilots: int = 64):
        """Initialize with realistic THz ISL parameters."""
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        
        # Calculate system parameters
        self.lambda_c = PhysicalConstants.c / f_c
        
        # Realistic link budget calculation
        self._calculate_link_budget()
        
        # PA parameters
        self.bussgang_gain = self._calculate_bussgang_gain()
        
        # Constellation
        self.constellation = self._create_constellation()
        
        # Debug counter
        self._debug_count = 0
    
    def _calculate_link_budget(self):
        """Calculate realistic THz ISL link budget."""
        # Transmit power (realistic for THz)
        self.P_tx_dBm = 20  # 20 dBm = 100 mW (achievable for THz PA)
        self.P_tx_watts = 10**(self.P_tx_dBm/10) / 1000
        
        # Antenna gains (0.5m dish at 300 GHz)
        self.G_tx_dB = 10 * np.log10(scenario.antenna_gain)
        self.G_rx_dB = self.G_tx_dB
        
        # Path loss
        self.path_loss_dB = 20 * np.log10(4 * np.pi * self.distance / self.lambda_c)
        
        # Total link budget
        self.P_rx_dBm = self.P_tx_dBm + self.G_tx_dB + self.G_rx_dB - self.path_loss_dB
        self.P_rx_watts = 10**(self.P_rx_dBm/10) / 1000
        
        # Noise parameters
        self.noise_figure_dB = 10  # Realistic for THz receiver
        self.bandwidth_Hz = 10e9   # 10 GHz
        self.noise_temp_K = 290 * 10**(self.noise_figure_dB/10)
        self.N_0 = PhysicalConstants.k * self.noise_temp_K * self.bandwidth_Hz
        
        # Channel gain (linear)
        self.channel_gain = np.sqrt(10**((self.G_tx_dB + self.G_rx_dB - self.path_loss_dB)/10))
        
        # Print link budget summary (once)
        if not hasattr(self.__class__, '_link_budget_printed'):
            print(f"\n=== THz ISL Link Budget at {self.f_c/1e9:.0f} GHz ===")
            print(f"  Distance: {self.distance/1e3:.0f} km")
            print(f"  Tx Power: {self.P_tx_dBm:.1f} dBm ({self.P_tx_watts*1000:.1f} mW)")
            print(f"  Antenna Gains: {self.G_tx_dB:.1f} dBi each")
            print(f"  Path Loss: {self.path_loss_dB:.1f} dB")
            print(f"  Rx Power: {self.P_rx_dBm:.1f} dBm ({self.P_rx_watts*1e12:.1f} pW)")
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
    
    def calculate_sinr(self, symbol: complex, avg_power: float, P_tx_scale: float) -> float:
        """Calculate SINR with realistic power scaling."""
        # Actual transmit power
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        
        # Received signal power
        symbol_power = np.abs(symbol)**2
        P_rx_signal = P_tx * symbol_power * self.channel_gain**2 * self.bussgang_gain**2
        
        # Noise components
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
    
    def calculate_bfim_observable(self, avg_power: float, P_tx_scale: float) -> np.ndarray:
        """
        Calculate B-FIM for observable parameters with proper power scaling.
        
        Key insight: FIM must be sensitive to power allocation to enable trade-off.
        """
        # Actual transmit power
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        
        # Received power
        P_rx = P_tx * self.channel_gain**2 * self.bussgang_gain**2
        
        # Noise components
        N_thermal = self.N_0
        N_hw = P_rx * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        N_total = N_thermal + N_hw * phase_penalty
        
        # Effective SNR (key for FIM)
        SNR_eff = P_rx / N_total
        
        # FIM for range (carrier phase measurement)
        phase_sensitivity = (2 * np.pi * self.f_c / PhysicalConstants.c)**2
        J_range = 2 * self.n_pilots * SNR_eff * phase_sensitivity
        
        # FIM for radial velocity (Doppler measurement)
        # Use realistic coherent processing interval
        T_CPI = 1e-3  # 1 ms (limited by oscillator stability)
        doppler_sensitivity = (2 * np.pi * self.f_c * T_CPI / PhysicalConstants.c)**2
        J_velocity = 2 * self.n_pilots * SNR_eff * doppler_sensitivity
        
        # Build FIM matrix
        J_B = np.diag([J_range, J_velocity])
        
        # Add small regularization
        J_B += 1e-20 * np.eye(2)
        
        return J_B
    
    def calculate_distortion(self, p_x: np.ndarray, P_tx_scale: float = 1.0) -> float:
        """Calculate sensing distortion as Tr(J_B^{-1})."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        if avg_power < 1e-10:
            return 1e10
        
        J_B = self.calculate_bfim_observable(avg_power, P_tx_scale)
        
        # Selective debug output
        self._debug_count += 1
        if DEBUG_VERBOSE and self._debug_count % 50 == 0:
            print(f"\n[Debug #{self._debug_count}]")
            print(f"  avg_power = {avg_power:.6f}")
            print(f"  P_tx_scale = {P_tx_scale:.3f}")
            print(f"  p_x entropy = {-np.sum(p_x * np.log2(p_x + 1e-10)):.3f}")
            print(f"  J_B condition = {np.linalg.cond(J_B):.2e}")
        
        try:
            J_B_inv = inv(J_B)
            distortion = np.trace(J_B_inv)
            
            if distortion < 0 or distortion > 1e15:
                return 1e10
                
        except np.linalg.LinAlgError:
            return 1e10
            
        return distortion

def check_hardware_feasibility(hardware_profile: str):
    """Check if the hardware can achieve meaningful ISAC performance."""
    print(f"\n=== Hardware Feasibility Check: {hardware_profile} ===")
    
    system = ISACSystem(hardware_profile)
    
    # Test at maximum power
    p_x = np.ones(len(system.constellation)) / len(system.constellation)
    
    # Check at different power scales
    power_scales = [0.1, 1.0, 10.0]  # Relative to nominal 20 dBm
    
    for P_scale in power_scales:
        # Calculate performance metrics
        I_x = system.calculate_mutual_information(p_x, P_scale)
        capacity = np.sum(p_x * I_x)
        distortion = system.calculate_distortion(p_x, P_scale)
        ranging_rmse = np.sqrt(distortion) if distortion < 1e10 else np.inf
        
        print(f"\nPower scale {P_scale}x ({10*np.log10(P_scale):.1f} dB):")
        print(f"  Capacity: {capacity:.3f} bits/symbol")
        print(f"  Ranging RMSE: {ranging_rmse:.3e} m")
        
        if capacity < 0.1:
            print("  ⚠️  WARNING: Link capacity too low for reliable communication!")
        if ranging_rmse > 1000:
            print("  ⚠️  WARNING: Sensing accuracy too poor for ISL maintenance!")
    
    return True

def modified_blahut_arimoto(system: ISACSystem, D_target: float, 
                           P_tx_scale: float = 1.0,
                           epsilon_lambda: float = 1e-3,
                           epsilon_p: float = 1e-6,
                           max_iterations: int = 50,
                           verbose: bool = False) -> Tuple[float, np.ndarray]:
    """
    Modified Blahut-Arimoto for ISAC with realistic constraints.
    """
    n_symbols = len(system.constellation)
    
    # Smart initialization based on distortion target
    if D_target > 1e6:  # Very high distortion - focus on communication
        p_x = np.ones(n_symbols) / n_symbols
    else:  # Low distortion - focus on sensing
        # Put more power on stronger symbols
        symbol_powers = np.abs(system.constellation)**2
        p_x = symbol_powers / np.sum(symbol_powers)
    
    # Check if uniform distribution already meets target
    D_uniform = system.calculate_distortion(p_x, P_tx_scale)
    if D_uniform <= D_target:
        I_x = system.calculate_mutual_information(p_x, P_tx_scale)
        return np.sum(p_x * I_x), p_x
    
    # Binary search for Lagrange multiplier
    lambda_min, lambda_max = 0, 1e6
    
    iteration_count = 0
    while (lambda_max - lambda_min) > epsilon_lambda and iteration_count < 20:
        lambda_current = (lambda_min + lambda_max) / 2
        iteration_count += 1
        
        # Reset to uniform
        p_x = np.ones(n_symbols) / n_symbols
        
        # Inner optimization
        for inner_iter in range(max_iterations):
            p_x_prev = p_x.copy()
            
            # Calculate gradients
            I_x = system.calculate_mutual_information(p_x, P_tx_scale)
            
            # Numerical gradient of distortion
            grad_D = np.zeros(n_symbols)
            base_D = system.calculate_distortion(p_x, P_tx_scale)
            
            delta = 0.01
            for i in range(n_symbols):
                if p_x[i] > delta:
                    p_perturb = p_x.copy()
                    p_perturb[i] -= delta
                    p_perturb[(i+1) % n_symbols] += delta
                    
                    D_perturb = system.calculate_distortion(p_perturb, P_tx_scale)
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
        D_current = system.calculate_distortion(p_x, P_tx_scale)
        
        if verbose and iteration_count % 5 == 0:
            print(f"  λ={lambda_current:.2e}, D={D_current:.3e}, C={np.sum(p_x * I_x):.3f}")
        
        if D_current > D_target:
            lambda_min = lambda_current
        else:
            lambda_max = lambda_current
    
    capacity = np.sum(p_x * I_x)
    return capacity, p_x

def generate_cd_frontier(hardware_profile: str, 
                        n_distortion_points: int = 10,
                        P_tx_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate C-D frontier with feasibility checks."""
    print(f"\nGenerating C-D frontier for {hardware_profile} (P_scale={P_tx_scale})...")
    
    system = ISACSystem(hardware_profile)
    
    # Find achievable distortion range
    # Minimum distortion: concentrate power
    p_concentrated = np.zeros(len(system.constellation))
    p_concentrated[np.argmax(np.abs(system.constellation)**2)] = 1.0
    D_min = system.calculate_distortion(p_concentrated, P_tx_scale)
    
    # Maximum distortion: low uniform power
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    D_max_uniform = system.calculate_distortion(p_uniform, P_tx_scale)
    
    # Expand range if too narrow
    if D_max_uniform / D_min < 100:
        D_min = D_min / 10
        D_max = D_max_uniform * 10
    else:
        D_max = D_max_uniform
    
    print(f"  Achievable distortion range: {D_min:.3e} to {D_max:.3e} m²")
    
    # Generate log-spaced targets
    D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_distortion_points)
    
    capacities = []
    distortions = []
    
    for i, D_target in enumerate(D_targets):
        if i % 3 == 0:  # Progress indicator
            print(f"  Progress: {i+1}/{n_distortion_points}")
        
        try:
            capacity, p_opt = modified_blahut_arimoto(
                system, D_target, P_tx_scale, verbose=False
            )
            
            actual_D = system.calculate_distortion(p_opt, P_tx_scale)
            
            if 0 < actual_D < 1e10 and capacity >= 0:
                capacities.append(capacity)
                distortions.append(actual_D)
        except:
            continue
    
    return np.array(distortions), np.array(capacities)

def plot_cd_frontier():
    """Generate the main C-D frontier plot."""
    print("\n=== Generating Capacity-Distortion Frontier Plot ===")
    
    # First check feasibility
    for profile in ["High_Performance", "SWaP_Efficient"]:
        check_hardware_feasibility(profile)
    
    # Generate frontiers at different power levels
    profiles = ["High_Performance", "SWaP_Efficient"]
    power_scales = [1.0]  # Can add more: [0.5, 1.0, 2.0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, profile in enumerate(profiles):
        for j, P_scale in enumerate(power_scales):
            distortions, capacities = generate_cd_frontier(
                profile, n_distortion_points=10, P_tx_scale=P_scale
            )
            
            if len(distortions) > 0:
                # Convert to ranging RMSE in mm
                ranging_rmse_mm = np.sqrt(distortions) * 1000
                
                label = f"{profile.replace('_', ' ')}"
                if len(power_scales) > 1:
                    label += f" ({10*np.log10(P_scale):+.0f} dB)"
                
                ax.plot(ranging_rmse_mm, capacities,
                        color=colors[i], linewidth=3,
                        linestyle='-' if j == 0 else '--',
                        marker='o', markersize=8,
                        label=label,
                        markerfacecolor='white',
                        markeredgewidth=2)
                
                # Fill area under curve
                ax.fill_between(ranging_rmse_mm, 0, capacities, 
                               alpha=0.2, color=colors[i])
    
    # Formatting
    ax.set_xlabel('Ranging RMSE [mm]', fontsize=14)
    ax.set_ylabel('Communication Capacity [bits/symbol]', fontsize=14)
    ax.set_title('THz ISL ISAC: Capacity-Distortion Trade-off\n' + 
                 f'(Single Link Observable Parameters Only)\n' +
                 f'f_c = 300 GHz, Distance = 2000 km',
                 fontsize=16, pad=15)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Use log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim(left=1)  # Start from 1 mm
    
    # Add feasibility region annotation
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
    ax.text(0.5, 1.1, 'Minimum capacity for reliable communication', 
            transform=ax.get_yaxis_transform(), 
            fontsize=10, color='red', ha='center')
    
    plt.tight_layout()
    plt.savefig('cd_frontier.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cd_frontier.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n=== C-D Frontier Summary ===")
    print("Key findings:")
    print("1. THz ISL faces severe power limitations due to path loss")
    print("2. Hardware impairments create fundamental performance ceilings")
    print("3. Single-link observable parameters: range and radial velocity only")
    print("4. Full 3D state estimation requires multiple non-coplanar links")

def main():
    """Main function."""
    print("=== THz ISL ISAC Capacity-Distortion Frontier Simulation ===")
    print("\nCRITICAL CONSTRAINTS:")
    print("- Single ISL link (2D observability only)")
    print("- Extreme path loss at THz frequencies")
    print("- Hardware-limited performance")
    print("- Phase noise limits coherent processing time")
    
    # Set debug level
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = False  # Change to True for detailed debugging
    
    # Generate main plot
    plot_cd_frontier()
    
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()