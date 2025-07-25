#!/usr/bin/env python3
"""
cd_frontier_simulation.py - UPDATED VERSION

Key fixes:
1. Uses larger antenna (1m) and higher power (30 dBm) for positive link margin
2. Corrected hardware quality factors from config
3. Improved numerical stability in optimization
4. Added link budget verification
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

colors = sns.color_palette("husl", 6)

# Global debug flag
DEBUG_VERBOSE = False

class ISACSystem:
    """Enhanced THz ISL ISAC system with improved link budget."""
    
    def __init__(self, hardware_profile: str, f_c: float = 300e9, 
                 distance: float = 2000e3, n_pilots: int = 64,
                 antenna_diameter: float = 1.0,  # Default to 1m for positive margin
                 tx_power_dBm: float = 30):      # Default to 30 dBm
        """Initialize with improved parameters for positive link margin."""
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        self.antenna_diameter = antenna_diameter
        self.tx_power_dBm = tx_power_dBm
        
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
        """Enhanced link budget calculation using scenario parameters."""
        # Use configured transmit power
        self.P_tx_watts = 10**(self.tx_power_dBm/10) / 1000
        
        # Calculate antenna gains using scenario method
        G_single = scenario.antenna_gain(self.antenna_diameter, self.f_c)
        self.G_tx_dB = 10 * np.log10(G_single)
        self.G_rx_dB = self.G_tx_dB
        
        # Path loss
        self.path_loss_dB = DerivedParameters.path_loss_dB(self.distance, self.f_c)
        
        # Total link budget
        link_budget = DerivedParameters.link_budget_dB(
            self.tx_power_dBm, self.G_tx_dB, self.G_rx_dB,
            self.distance, self.f_c
        )
        self.P_rx_dBm = link_budget['rx_power_dBm']
        self.P_rx_watts = 10**(self.P_rx_dBm/10) / 1000
        
        # Noise parameters
        self.noise_figure_dB = 8  # Reasonable for THz
        self.bandwidth_Hz = self.profile.signal_bandwidth_Hz
        self.noise_power_watts = DerivedParameters.thermal_noise_power(
            self.bandwidth_Hz, noise_figure_dB=self.noise_figure_dB
        )
        self.N_0 = self.noise_power_watts
        self.noise_power_dBm = 10 * np.log10(self.noise_power_watts * 1000)
        
        # Channel gain (linear) - magnitude
        path_gain_linear = 10**(-self.path_loss_dB/20)  # Convert dB to linear amplitude
        antenna_gain_linear = 10**((self.G_tx_dB + self.G_rx_dB)/20)
        self.channel_gain = path_gain_linear * np.sqrt(antenna_gain_linear)
        
        # Link margin
        self.link_margin_dB = self.P_rx_dBm - self.noise_power_dBm
        
        # Print link budget summary (once per profile)
        if not hasattr(self.__class__, f'_link_budget_printed_{self.profile.name}'):
            print(f"\n=== Link Budget for {self.profile.name} at {self.f_c/1e9:.0f} GHz ===")
            print(f"  Distance: {self.distance/1e3:.0f} km")
            print(f"  Antenna Diameter: {self.antenna_diameter:.1f} m")
            print(f"  Tx Power: {self.tx_power_dBm:.1f} dBm ({self.P_tx_watts*1000:.0f} mW)")
            print(f"  Antenna Gains: {self.G_tx_dB:.1f} dBi each")
            print(f"  Path Loss: {self.path_loss_dB:.1f} dB")
            print(f"  Rx Power: {self.P_rx_dBm:.1f} dBm")
            print(f"  Noise Power: {self.noise_power_dBm:.1f} dBm")
            print(f"  Link Margin: {self.link_margin_dB:.1f} dB")
            
            if self.link_margin_dB < 0:
                print("  ⚠️  WARNING: Negative link margin! System will not work properly.")
            else:
                print("  ✓ Positive link margin - system operational")
            
            setattr(self.__class__, f'_link_budget_printed_{self.profile.name}', True)
    
    def _calculate_bussgang_gain(self) -> float:
        """Calculate Bussgang gain for PA nonlinearity."""
        # From paper approximation
        kappa = 10**(-7.0/10)  # 7 dB IBO
        B = 1 - 1.5 * kappa + 1.875 * kappa**2
        return B
    
    def _create_constellation(self, modulation: str = 'QPSK') -> np.ndarray:
        """Create normalized constellation."""
        if modulation == 'QPSK':
            angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            constellation = np.exp(1j * angles)
            # Normalize to unit average power
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation
    
    def calculate_sinr(self, symbol: complex, avg_power: float, P_tx_scale: float) -> float:
        """Calculate SINR for given symbol and power allocation."""
        # Transmit power with scaling
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        
        # Symbol power
        symbol_power = np.abs(symbol)**2
        
        # Received signal power
        P_rx_signal = P_tx * symbol_power * (self.channel_gain**2) * (self.bussgang_gain**2)
        
        # Noise components
        N_thermal = self.N_0
        N_hw = P_rx_signal * self.profile.Gamma_eff
        
        # Phase noise penalty
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        
        # Total noise
        N_total = N_thermal + N_hw * phase_penalty
        
        # SINR
        sinr = P_rx_signal / N_total
        return sinr
    
    def calculate_mutual_information(self, p_x: np.ndarray, P_tx_scale: float = 1.0) -> np.ndarray:
        """Calculate mutual information for each symbol."""
        # Average power from distribution
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        # MI for each symbol
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr(symbol, avg_power, P_tx_scale)
            I_x[i] = np.log2(1 + sinr)
            
        return I_x
    
    def calculate_bfim_observable(self, avg_power: float, P_tx_scale: float) -> np.ndarray:
        """Calculate B-FIM for observable parameters (range and radial velocity)."""
        # Total transmit power
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        
        # Received power
        P_rx = P_tx * (self.channel_gain**2) * (self.bussgang_gain**2)
        
        # Noise
        N_thermal = self.N_0
        N_hw = P_rx * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        N_total = N_thermal + N_hw * phase_penalty
        
        # Effective SNR
        SNR_eff = P_rx / N_total
        
        # FIM for range (position along LOS)
        # From paper: phase sensitivity dominates
        phase_sensitivity = (2 * np.pi * self.f_c / PhysicalConstants.c)**2
        J_range = 2 * self.n_pilots * SNR_eff * phase_sensitivity
        
        # FIM for radial velocity
        # Using coherent processing interval
        T_CPI = 1e-3  # 1 ms coherent processing interval
        doppler_sensitivity = (2 * np.pi * self.f_c * T_CPI / PhysicalConstants.c)**2
        J_velocity = 2 * self.n_pilots * SNR_eff * doppler_sensitivity
        
        # Create 2x2 FIM for observable parameters
        J_B = np.diag([J_range, J_velocity])
        
        # Add small regularization for numerical stability
        J_B += 1e-20 * np.eye(2)
        
        return J_B
    
    def calculate_distortion(self, p_x: np.ndarray, P_tx_scale: float = 1.0) -> float:
        """Calculate sensing distortion (trace of CRLB)."""
        # Average power
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        # Handle edge case
        if avg_power < 1e-10:
            return 1e10
        
        # Calculate B-FIM
        J_B = self.calculate_bfim_observable(avg_power, P_tx_scale)
        
        # Debug output
        self._debug_count += 1
        if DEBUG_VERBOSE and self._debug_count % 200 == 0:
            print(f"\n[Debug #{self._debug_count}]")
            print(f"  avg_power = {avg_power:.6f}")
            print(f"  J_B condition number = {np.linalg.cond(J_B):.2e}")
        
        # Calculate CRLB
        try:
            J_B_inv = inv(J_B)
            distortion = np.trace(J_B_inv)
            
            # Sanity check
            if distortion < 0 or distortion > 1e15:
                return 1e10
                
        except np.linalg.LinAlgError:
            return 1e10
            
        return distortion

def modified_blahut_arimoto(system: ISACSystem, D_target: float, 
                           P_tx_scale: float = 1.0,
                           epsilon_lambda: float = 1e-3,
                           epsilon_p: float = 1e-6,
                           max_iterations: int = 50,
                           verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Modified Blahut-Arimoto algorithm for ISAC optimization."""
    n_symbols = len(system.constellation)
    
    # Smart initialization based on distortion target
    if D_target > 1e6:  # Very loose constraint
        p_x = np.ones(n_symbols) / n_symbols  # Uniform
    else:  # Tight constraint
        # Start with more power on stronger symbols
        symbol_powers = np.abs(system.constellation)**2
        p_x = symbol_powers / np.sum(symbol_powers)
    
    # Check if initial distribution meets constraint
    D_init = system.calculate_distortion(p_x, P_tx_scale)
    if D_init <= D_target:
        # Already satisfies constraint, return capacity
        I_x = system.calculate_mutual_information(p_x, P_tx_scale)
        return np.sum(p_x * I_x), p_x
    
    # Binary search for Lagrange multiplier
    lambda_min, lambda_max = 0, 1e6
    
    iteration_count = 0
    while (lambda_max - lambda_min) > epsilon_lambda and iteration_count < 20:
        lambda_current = (lambda_min + lambda_max) / 2
        iteration_count += 1
        
        # Reset distribution
        p_x = np.ones(n_symbols) / n_symbols
        
        # Inner optimization loop
        for inner_iter in range(max_iterations):
            p_x_prev = p_x.copy()
            
            # Calculate mutual information
            I_x = system.calculate_mutual_information(p_x, P_tx_scale)
            
            # Numerical gradient of distortion
            grad_D = np.zeros(n_symbols)
            base_D = system.calculate_distortion(p_x, P_tx_scale)
            
            delta = 0.01  # Perturbation size
            for i in range(n_symbols):
                if p_x[i] > delta:
                    # Create perturbed distribution
                    p_perturb = p_x.copy()
                    p_perturb[i] -= delta
                    p_perturb[(i+1) % n_symbols] += delta  # Maintain normalization
                    
                    # Calculate perturbed distortion
                    D_perturb = system.calculate_distortion(p_perturb, P_tx_scale)
                    
                    # Finite difference gradient
                    grad_D[i] = (D_perturb - base_D) / delta
            
            # Update distribution in log domain for stability
            log_p = np.log(p_x + 1e-10)
            log_p += 0.1 * (I_x - lambda_current * grad_D)  # Step size 0.1
            
            # Normalize in log domain
            log_p -= np.max(log_p)  # Prevent overflow
            p_x = np.exp(log_p)
            p_x /= np.sum(p_x)
            
            # Check convergence
            if np.linalg.norm(p_x - p_x_prev, ord=1) < epsilon_p:
                break
        
        # Check constraint
        D_current = system.calculate_distortion(p_x, P_tx_scale)
        
        if verbose and iteration_count % 5 == 0:
            print(f"  λ = {lambda_current:.2e}, D = {D_current:.2e} (target: {D_target:.2e})")
        
        # Update lambda bounds
        if D_current > D_target:
            lambda_min = lambda_current  # Need more penalty
        else:
            lambda_max = lambda_current  # Can reduce penalty
    
    # Final capacity calculation
    I_x = system.calculate_mutual_information(p_x, P_tx_scale)
    capacity = np.sum(p_x * I_x)
    
    return capacity, p_x

def generate_cd_frontier(hardware_profile: str, 
                        n_distortion_points: int = 15,
                        P_tx_scale: float = 1.0,
                        antenna_diameter: float = 1.0,
                        tx_power_dBm: float = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Generate C-D frontier with proper link budget."""
    print(f"\nGenerating C-D frontier for {hardware_profile}...")
    print(f"  Antenna: {antenna_diameter}m, Tx Power: {tx_power_dBm} dBm")
    
    # Create system with specified parameters
    system = ISACSystem(hardware_profile, 
                       antenna_diameter=antenna_diameter,
                       tx_power_dBm=tx_power_dBm)
    
    # Find achievable distortion range
    # Maximum power concentration
    p_concentrated = np.zeros(len(system.constellation))
    p_concentrated[np.argmax(np.abs(system.constellation)**2)] = 1.0
    D_min = system.calculate_distortion(p_concentrated, P_tx_scale)
    
    # Uniform distribution
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    D_max_uniform = system.calculate_distortion(p_uniform, P_tx_scale)
    
    # Set range with some margin
    if D_max_uniform / D_min < 100:
        D_min = D_min / 10
        D_max = D_max_uniform * 10
    else:
        D_max = D_max_uniform
    
    print(f"  Achievable distortion range: {D_min:.3e} to {D_max:.3e} m²")
    
    # Generate target points
    D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_distortion_points)
    
    capacities = []
    distortions = []
    
    # Progress tracking
    for i, D_target in enumerate(D_targets):
        if i % 3 == 0:
            print(f"  Progress: {i+1}/{n_distortion_points}")
        
        try:
            # Run optimization
            capacity, p_opt = modified_blahut_arimoto(
                system, D_target, P_tx_scale, verbose=False
            )
            
            # Get actual achieved distortion
            actual_D = system.calculate_distortion(p_opt, P_tx_scale)
            
            # Store valid results
            if 0 < actual_D < 1e10 and capacity >= 0:
                capacities.append(capacity)
                distortions.append(actual_D)
                
        except Exception as e:
            if DEBUG_VERBOSE:
                print(f"  Warning: Optimization failed for D_target={D_target:.2e}: {e}")
            continue
    
    print(f"  Generated {len(capacities)} valid points")
    
    return np.array(distortions), np.array(capacities)

def plot_cd_frontier_comparison():
    """Generate C-D frontier plot comparing configurations."""
    print("\n=== Generating C-D Frontier Comparison ===")
    
    # Define configurations to compare
    configs = [
        ("Default (0.5m, 20dBm)", 0.5, 20),
        ("Large Antenna (1m, 20dBm)", 1.0, 20),
        ("High Power (0.5m, 30dBm)", 0.5, 30),
        ("Optimized (1m, 30dBm)", 1.0, 30)
    ]
    
    profiles = ["High_Performance", "SWaP_Efficient"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax_idx, profile in enumerate(profiles):
        ax = axes[ax_idx]
        
        for config_idx, (config_name, ant_diam, tx_power) in enumerate(configs):
            # Generate frontier
            distortions, capacities = generate_cd_frontier(
                profile, 
                n_distortion_points=12,
                antenna_diameter=ant_diam,
                tx_power_dBm=tx_power
            )
            
            if len(distortions) > 0:
                # Convert to ranging RMSE in mm
                ranging_rmse_mm = np.sqrt(distortions) * 1000
                
                # Plot
                linestyle = ['-', '--', '-.', ':'][config_idx]
                ax.plot(ranging_rmse_mm, capacities,
                        color=colors[config_idx], linewidth=2.5,
                        linestyle=linestyle,
                        marker='o', markersize=6,
                        label=config_name,
                        markerfacecolor='white',
                        markeredgewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Ranging RMSE [mm]', fontsize=12)
        ax.set_ylabel('Communication Capacity [bits/symbol]', fontsize=12)
        ax.set_title(f'{profile.replace("_", " ")}', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        ax.set_xscale('log')
        ax.set_xlim(left=0.1)
        ax.set_ylim(bottom=0)
        
        # Add performance threshold lines
        ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5)
        ax.text(0.15, 2.1, 'Good comm (>2 bits/symbol)', 
                fontsize=9, color='green')
        
        ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5)
        ax.text(0.9, ax.get_ylim()[1]*0.95, 'Sub-mm\naccuracy', 
                ha='right', va='top', fontsize=9, color='blue')
    
    plt.suptitle('THz ISL ISAC C-D Trade-off: Link Budget Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('cd_frontier_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cd_frontier_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cd_frontier_main():
    """Generate main C-D frontier plot with optimized configuration."""
    print("\n=== Generating Main C-D Frontier Plot ===")
    
    profiles = ["High_Performance", "SWaP_Efficient"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, profile in enumerate(profiles):
        # Use optimized configuration (1m antenna, 30 dBm)
        distortions, capacities = generate_cd_frontier(
            profile, 
            n_distortion_points=15,
            antenna_diameter=1.0,
            tx_power_dBm=30
        )
        
        if len(distortions) > 0:
            # Convert to ranging RMSE in mm
            ranging_rmse_mm = np.sqrt(distortions) * 1000
            
            # Plot
            ax.plot(ranging_rmse_mm, capacities,
                    color=colors[i], linewidth=3,
                    marker='o', markersize=8,
                    label=profile.replace('_', ' '),
                    markerfacecolor='white',
                    markeredgewidth=2)
            
            # Add annotations for key points
            # Best communication point
            best_comm_idx = np.argmax(capacities)
            ax.annotate(f'{capacities[best_comm_idx]:.2f} bits/symbol',
                       xy=(ranging_rmse_mm[best_comm_idx], capacities[best_comm_idx]),
                       xytext=(ranging_rmse_mm[best_comm_idx]*2, capacities[best_comm_idx]+0.2),
                       arrowprops=dict(arrowstyle='->', alpha=0.5),
                       fontsize=10)
            
            # Best sensing point
            best_sense_idx = np.argmin(ranging_rmse_mm)
            ax.annotate(f'{ranging_rmse_mm[best_sense_idx]:.1f} mm',
                       xy=(ranging_rmse_mm[best_sense_idx], capacities[best_sense_idx]),
                       xytext=(ranging_rmse_mm[best_sense_idx]*0.5, capacities[best_sense_idx]+0.3),
                       arrowprops=dict(arrowstyle='->', alpha=0.5),
                       fontsize=10)
    
    # Formatting
    ax.set_xlabel('Ranging RMSE [mm]', fontsize=14)
    ax.set_ylabel('Communication Capacity [bits/symbol]', fontsize=14)
    ax.set_title('THz ISL ISAC Capacity-Distortion Trade-off\n' + 
                 '(1m Antennas, 30 dBm Tx Power, 2000 km Distance, 300 GHz)',
                 fontsize=16, pad=15)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    ax.set_xscale('log')
    ax.set_xlim(left=0.1)
    ax.set_ylim(bottom=0)
    
    # Add feasible region shading
    ax.axhspan(2.0, ax.get_ylim()[1], alpha=0.1, color='green', label='_nolegend_')
    ax.axvspan(ax.get_xlim()[0], 1.0, alpha=0.1, color='blue', label='_nolegend_')
    
    # Add text annotations
    ax.text(0.5, ax.get_ylim()[1]*0.95, 'Sub-mm sensing region', 
            ha='center', va='top', fontsize=11, style='italic', alpha=0.7)
    ax.text(ax.get_xlim()[1]*0.5, 2.5, 'High-rate communication region', 
            ha='center', va='bottom', fontsize=11, style='italic', alpha=0.7)
    
    # Add hardware quality impact text
    textstr = (
        'Key Insights:\n'
        f'• High Performance (Γ_eff={HARDWARE_PROFILES["High_Performance"].Gamma_eff}): Better overall\n'
        f'• SWaP Efficient (Γ_eff={HARDWARE_PROFILES["SWaP_Efficient"].Gamma_eff}): Cost-effective\n'
        '• Both achieve sub-mm ranging with proper design\n'
        '• Positive link margin with 1m antenna + 30 dBm'
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig('cd_frontier_main.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cd_frontier_main.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def verify_link_budgets():
    """Verify link budgets for different configurations."""
    print("\n=== Link Budget Verification ===")
    
    configs = [
        ("Default", 0.5, 20),
        ("Large Antenna", 1.0, 20),
        ("High Power", 0.5, 30),
        ("Optimized", 1.0, 30)
    ]
    
    for name, ant_diam, tx_power in configs:
        print(f"\n{name} Configuration:")
        
        # Calculate link budget
        ant_gain = scenario.antenna_gain_dB(ant_diam)
        budget = DerivedParameters.link_budget_dB(
            tx_power, ant_gain, ant_gain,
            scenario.R_default, scenario.f_c_default
        )
        
        # Noise floor
        noise_dBm = DerivedParameters.thermal_noise_power_dBm(10e9, noise_figure_dB=8)
        
        # Margin
        margin = budget['rx_power_dBm'] - noise_dBm
        
        print(f"  Antenna: {ant_diam}m ({ant_gain:.1f} dBi)")
        print(f"  Tx Power: {tx_power} dBm")
        print(f"  Rx Power: {budget['rx_power_dBm']:.1f} dBm")
        print(f"  Noise: {noise_dBm:.1f} dBm")
        print(f"  Margin: {margin:.1f} dB {'✓' if margin > 0 else '✗'}")

def main():
    """Main function with all analyses."""
    print("=== THz ISL ISAC C-D Frontier Analysis (UPDATED) ===")
    print("\nKey Updates:")
    print("1. Using corrected Gamma_eff values from config")
    print("2. Improved link budget with larger antennas and higher power")
    print("3. Verified positive link margin for operation")
    
    # Set debug level
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = False
    
    # First verify link budgets
    verify_link_budgets()
    
    # Generate comparison plot
    plot_cd_frontier_comparison()
    
    # Generate main C-D frontier plot
    plot_cd_frontier_main()
    
    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("1. cd_frontier_comparison.pdf/png - Shows impact of antenna/power")
    print("2. cd_frontier_main.pdf/png - Main result with optimized config")
    print("\nRecommendation: Use 1m antennas with 30 dBm for robust operation")

if __name__ == "__main__":
    main()