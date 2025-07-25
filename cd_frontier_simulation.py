#!/usr/bin/env python3
"""
cd_frontier_simulation.py - Enhanced Version with Additional Analyses

Key improvements:
1. Realistic link budget with positive margin
2. SNR scan to find hardware limits
3. Hardware parameter sensitivity analysis
4. Multiple visualization options
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
                 antenna_diameter: float = 1.0):  # Increased to 1m
        """Initialize with improved parameters."""
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
        self.P_tx_dBm = 30  # 30 dBm = 1W (realistic for satellite PA)
        self.P_tx_watts = 10**(self.P_tx_dBm/10) / 1000
        
        # Enhanced antenna gains with larger diameter
        antenna_efficiency = 0.65  # Typical for large reflector
        G_single = antenna_efficiency * (np.pi * self.antenna_diameter / self.lambda_c)**2
        self.G_tx_dB = 10 * np.log10(G_single)
        self.G_rx_dB = self.G_tx_dB
        
        # Path loss
        self.path_loss_dB = 20 * np.log10(4 * np.pi * self.distance / self.lambda_c)
        
        # Total link budget
        self.P_rx_dBm = self.P_tx_dBm + self.G_tx_dB + self.G_rx_dB - self.path_loss_dB
        self.P_rx_watts = 10**(self.P_rx_dBm/10) / 1000
        
        # Noise parameters
        self.noise_figure_dB = 8  # Improved NF for better LNA
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
    
    def calculate_sinr(self, symbol: complex, avg_power: float, P_tx_scale: float) -> float:
        """Calculate SINR with realistic power scaling."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        symbol_power = np.abs(symbol)**2
        P_rx_signal = P_tx * symbol_power * self.channel_gain**2 * self.bussgang_gain**2
        
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
        """Calculate B-FIM for observable parameters."""
        P_tx = self.P_tx_watts * P_tx_scale * avg_power
        P_rx = P_tx * self.channel_gain**2 * self.bussgang_gain**2
        
        N_thermal = self.N_0
        N_hw = P_rx * self.profile.Gamma_eff
        phase_penalty = np.exp(self.profile.phase_noise_variance)
        N_total = N_thermal + N_hw * phase_penalty
        
        SNR_eff = P_rx / N_total
        
        # FIM for range
        phase_sensitivity = (2 * np.pi * self.f_c / PhysicalConstants.c)**2
        J_range = 2 * self.n_pilots * SNR_eff * phase_sensitivity
        
        # FIM for radial velocity
        T_CPI = 1e-3  # 1 ms
        doppler_sensitivity = (2 * np.pi * self.f_c * T_CPI / PhysicalConstants.c)**2
        J_velocity = 2 * self.n_pilots * SNR_eff * doppler_sensitivity
        
        J_B = np.diag([J_range, J_velocity])
        J_B += 1e-20 * np.eye(2)
        
        return J_B
    
    def calculate_distortion(self, p_x: np.ndarray, P_tx_scale: float = 1.0) -> float:
        """Calculate sensing distortion."""
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        if avg_power < 1e-10:
            return 1e10
        
        J_B = self.calculate_bfim_observable(avg_power, P_tx_scale)
        
        self._debug_count += 1
        if DEBUG_VERBOSE and self._debug_count % 50 == 0:
            print(f"\n[Debug #{self._debug_count}]")
            print(f"  avg_power = {avg_power:.6f}")
            print(f"  J_B condition = {np.linalg.cond(J_B):.2e}")
        
        try:
            J_B_inv = inv(J_B)
            distortion = np.trace(J_B_inv)
            
            if distortion < 0 or distortion > 1e15:
                return 1e10
                
        except np.linalg.LinAlgError:
            return 1e10
            
        return distortion

def plot_snr_to_hardware_limit():
    """Plot SNR required to reach hardware-limited capacity ceiling."""
    print("\n=== Generating SNR to Hardware Limit Analysis ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SNR range from very low to very high
    snr_dB = np.linspace(-10, 60, 100)
    snr_linear = 10**(snr_dB/10)
    
    for idx, (profile_name, ax) in enumerate([(k, ax) for k, ax in zip(HARDWARE_PROFILES.keys(), [ax1, ax2])]):
        system = ISACSystem(profile_name)
        profile = HARDWARE_PROFILES[profile_name]
        
        # Calculate capacity for uniform distribution
        p_x = np.ones(len(system.constellation)) / len(system.constellation)
        capacities = []
        
        for snr in snr_linear:
            # Override the default SNR calculation
            I_x = []
            for symbol in system.constellation:
                # Direct SINR calculation
                P_rx = system.P_tx_watts * system.channel_gain**2 * system.bussgang_gain**2
                N_0 = P_rx / snr  # Back-calculate noise
                N_hw = P_rx * profile.Gamma_eff * np.exp(profile.phase_noise_variance)
                sinr_eff = P_rx / (N_0 + N_hw)
                I_x.append(np.log2(1 + sinr_eff))
            
            capacity = np.mean(I_x)
            capacities.append(capacity)
        
        capacities = np.array(capacities)
        
        # Calculate hardware ceiling
        ceiling = np.log2(1 + np.exp(-profile.phase_noise_variance) / profile.Gamma_eff)
        
        # Find SNR where capacity reaches 95% of ceiling
        idx_95 = np.argmin(np.abs(capacities - 0.95 * ceiling))
        snr_95_dB = snr_dB[idx_95]
        
        # Plot
        ax.plot(snr_dB, capacities, 'b-', linewidth=3, label='Actual Capacity')
        ax.axhline(y=ceiling, color='r', linestyle='--', linewidth=2, label=f'Hardware Ceiling: {ceiling:.2f} bits/symbol')
        ax.axhline(y=0.95*ceiling, color='g', linestyle=':', alpha=0.5)
        ax.axvline(x=snr_95_dB, color='g', linestyle=':', alpha=0.5)
        
        # Add annotation
        ax.annotate(f'95% of ceiling at\nSNR = {snr_95_dB:.1f} dB', 
                   xy=(snr_95_dB, 0.95*ceiling), 
                   xytext=(snr_95_dB-10, 0.8*ceiling),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   fontsize=10, ha='right')
        
        # Shade hardware-limited region
        ax.axvspan(snr_95_dB, 60, alpha=0.2, color='red', label='Hardware-limited region')
        
        ax.set_xlabel('SNR [dB]', fontsize=12)
        ax.set_ylabel('Capacity [bits/symbol]', fontsize=12)
        ax.set_title(f'{profile_name.replace("_", " ")}\n(Γ_eff = {profile.Gamma_eff})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xlim(-10, 60)
        ax.set_ylim(0, ceiling * 1.1)
    
    plt.suptitle('SNR Required to Reach Hardware-Limited Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('snr_to_hardware_limit.pdf', format='pdf', dpi=300)
    plt.savefig('snr_to_hardware_limit.png', format='png', dpi=300)
    plt.show()
    
    print(f"High Performance: 95% of ceiling at {snr_dB[np.argmin(np.abs(capacities - 0.95 * ceiling)):.1f} dB")

def plot_gamma_eff_sensitivity():
    """Plot system performance sensitivity to hardware quality factor."""
    print("\n=== Generating Hardware Quality Factor Sensitivity Analysis ===")
    
    # Gamma_eff range
    gamma_eff_values = np.logspace(-3, -1, 20)  # 0.001 to 0.1
    
    # Fixed conditions
    snr_dB = 30
    snr_linear = 10**(snr_dB/10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Store results
    capacity_results = []
    ranging_rmse_results = []
    
    for gamma_eff in gamma_eff_values:
        # Create temporary profile
        temp_profile = HARDWARE_PROFILES["High_Performance"]
        original_gamma = temp_profile.Gamma_eff
        temp_profile.Gamma_eff = gamma_eff
        
        # Create system
        system = ISACSystem("High_Performance")
        
        # Calculate capacity
        p_x = np.ones(len(system.constellation)) / len(system.constellation)
        I_x = system.calculate_mutual_information(p_x)
        capacity = np.mean(I_x)
        capacity_results.append(capacity)
        
        # Calculate CRLB
        distortion = system.calculate_distortion(p_x)
        ranging_rmse = np.sqrt(distortion) * 1000  # Convert to mm
        ranging_rmse_results.append(ranging_rmse)
        
        # Restore original value
        temp_profile.Gamma_eff = original_gamma
    
    # Plot capacity
    ax1.semilogx(gamma_eff_values, capacity_results, 'b-', linewidth=3, marker='o', markersize=8)
    ax1.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='High Performance')
    ax1.axvline(x=0.045, color='g', linestyle='--', alpha=0.5, label='SWaP Efficient')
    ax1.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=12)
    ax1.set_ylabel('Capacity [bits/symbol]', fontsize=12)
    ax1.set_title(f'Communication Performance\n(SNR = {snr_dB} dB)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot ranging RMSE
    ax2.loglog(gamma_eff_values, ranging_rmse_results, 'r-', linewidth=3, marker='s', markersize=8)
    ax2.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='High Performance')
    ax2.axvline(x=0.045, color='g', linestyle='--', alpha=0.5, label='SWaP Efficient')
    ax2.set_xlabel('Hardware Quality Factor Γ_eff', fontsize=12)
    ax2.set_ylabel('Ranging RMSE [mm]', fontsize=12)
    ax2.set_title(f'Sensing Performance\n(SNR = {snr_dB} dB)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('System Performance Sensitivity to Hardware Quality', fontsize=16)
    plt.tight_layout()
    plt.savefig('gamma_eff_sensitivity.pdf', format='pdf', dpi=300)
    plt.savefig('gamma_eff_sensitivity.png', format='png', dpi=300)
    plt.show()

def plot_3d_performance_landscape():
    """Plot 3D performance landscape over frequency and distance."""
    print("\n=== Generating 3D Performance Landscape ===")
    
    # Parameter ranges
    frequencies_GHz = np.linspace(100, 600, 15)
    distances_km = np.linspace(500, 5000, 15)
    
    # Create meshgrid
    F, D = np.meshgrid(frequencies_GHz, distances_km)
    
    # Calculate capacity for each point
    capacity_grid = np.zeros_like(F)
    
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            f_Hz = F[i, j] * 1e9
            d_m = D[i, j] * 1e3
            
            # Create system
            system = ISACSystem("High_Performance", f_c=f_Hz, distance=d_m)
            
            # Calculate capacity
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x)
            capacity_grid[i, j] = np.mean(I_x)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(F, D, capacity_grid, cmap='viridis', 
                          edgecolor='none', alpha=0.8)
    
    # Add contour lines at the bottom
    contours = ax.contour(F, D, capacity_grid, zdir='z', offset=0, 
                          cmap='viridis', alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Frequency [GHz]', fontsize=12)
    ax.set_ylabel('Distance [km]', fontsize=12)
    ax.set_zlabel('Capacity [bits/symbol]', fontsize=12)
    ax.set_title('THz ISL ISAC Performance Landscape', fontsize=16, pad=20)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('3d_performance_landscape.pdf', format='pdf', dpi=300)
    plt.savefig('3d_performance_landscape.png', format='png', dpi=300)
    plt.show()

def plot_isac_feasibility_regions():
    """Plot ISAC feasibility regions in parameter space."""
    print("\n=== Generating ISAC Feasibility Regions ===")
    
    # Parameter ranges
    tx_power_dBm = np.linspace(10, 40, 30)
    distances_km = np.linspace(500, 5000, 30)
    
    # Create meshgrid
    P, D = np.meshgrid(tx_power_dBm, distances_km)
    
    # Define feasibility criteria
    min_capacity = 1.0  # bits/symbol
    max_ranging_rmse = 10.0  # mm
    
    # Calculate feasibility for each point
    comm_feasible = np.zeros_like(P)
    sense_feasible = np.zeros_like(P)
    
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            # Create system with custom power
            system = ISACSystem("SWaP_Efficient", distance=D[i,j]*1e3)
            system.P_tx_dBm = P[i,j]
            system.P_tx_watts = 10**(P[i,j]/10) / 1000
            
            # Recalculate link budget
            system._calculate_enhanced_link_budget()
            
            # Check communication feasibility
            p_x = np.ones(len(system.constellation)) / len(system.constellation)
            I_x = system.calculate_mutual_information(p_x)
            capacity = np.mean(I_x)
            comm_feasible[i,j] = 1 if capacity >= min_capacity else 0
            
            # Check sensing feasibility
            distortion = system.calculate_distortion(p_x)
            ranging_rmse = np.sqrt(distortion) * 1000
            sense_feasible[i,j] = 1 if ranging_rmse <= max_ranging_rmse else 0
    
    # Combined feasibility
    isac_feasible = comm_feasible * sense_feasible
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    ax.set_xlabel('Transmit Power [dBm]', fontsize=12)
    ax.set_ylabel('ISL Distance [km]', fontsize=12)
    ax.set_title('ISAC Feasibility Regions\n(C ≥ 1 bit/symbol, RMSE ≤ 10 mm)', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Infeasible'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Communication Only'),
        Patch(facecolor='lightblue', edgecolor='black', label='Sensing Only'),
        Patch(facecolor='darkgreen', edgecolor='black', label='ISAC Feasible')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('isac_feasibility_regions.pdf', format='pdf', dpi=300)
    plt.savefig('isac_feasibility_regions.png', format='png', dpi=300)
    plt.show()

def modified_blahut_arimoto(system: ISACSystem, D_target: float, 
                           P_tx_scale: float = 1.0,
                           epsilon_lambda: float = 1e-3,
                           epsilon_p: float = 1e-6,
                           max_iterations: int = 50,
                           verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Modified Blahut-Arimoto for ISAC."""
    n_symbols = len(system.constellation)
    
    # Smart initialization
    if D_target > 1e6:
        p_x = np.ones(n_symbols) / n_symbols
    else:
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
        
        p_x = np.ones(n_symbols) / n_symbols
        
        # Inner optimization
        for inner_iter in range(max_iterations):
            p_x_prev = p_x.copy()
            
            I_x = system.calculate_mutual_information(p_x, P_tx_scale)
            
            # Numerical gradient
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
        
        if D_current > D_target:
            lambda_min = lambda_current
        else:
            lambda_max = lambda_current
    
    capacity = np.sum(p_x * I_x)
    return capacity, p_x

def generate_cd_frontier(hardware_profile: str, 
                        n_distortion_points: int = 10,
                        P_tx_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate C-D frontier."""
    print(f"\nGenerating C-D frontier for {hardware_profile} (P_scale={P_tx_scale})...")
    
    system = ISACSystem(hardware_profile)
    
    # Find achievable distortion range
    p_concentrated = np.zeros(len(system.constellation))
    p_concentrated[np.argmax(np.abs(system.constellation)**2)] = 1.0
    D_min = system.calculate_distortion(p_concentrated, P_tx_scale)
    
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    D_max_uniform = system.calculate_distortion(p_uniform, P_tx_scale)
    
    if D_max_uniform / D_min < 100:
        D_min = D_min / 10
        D_max = D_max_uniform * 10
    else:
        D_max = D_max_uniform
    
    print(f"  Achievable distortion range: {D_min:.3e} to {D_max:.3e} m²")
    
    D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_distortion_points)
    
    capacities = []
    distortions = []
    
    for i, D_target in enumerate(D_targets):
        if i % 3 == 0:
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

def plot_cd_frontier_enhanced():
    """Enhanced C-D frontier plot with multiple power levels."""
    print("\n=== Generating Enhanced C-D Frontier Plot ===")
    
    profiles = ["High_Performance", "SWaP_Efficient"]
    power_scales = [0.5, 1.0, 2.0]  # -3dB, 0dB, +3dB
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, profile in enumerate(profiles):
        for j, P_scale in enumerate(power_scales):
            distortions, capacities = generate_cd_frontier(
                profile, n_distortion_points=10, P_tx_scale=P_scale
            )
            
            if len(distortions) > 0:
                ranging_rmse_mm = np.sqrt(distortions) * 1000
                
                label = f"{profile.replace('_', ' ')}"
                if j == 0:
                    label += " (-3dB)"
                elif j == 2:
                    label += " (+3dB)"
                
                linestyle = ['-', '-', '--'][j]
                alpha = [0.6, 1.0, 0.6][j]
                
                ax.plot(ranging_rmse_mm, capacities,
                        color=colors[i], linewidth=2.5,
                        linestyle=linestyle, alpha=alpha,
                        marker='o', markersize=6,
                        label=label if j != 1 or i == 0 else profile.replace('_', ' '),
                        markerfacecolor='white',
                        markeredgewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Ranging RMSE [mm]', fontsize=14)
    ax.set_ylabel('Communication Capacity [bits/symbol]', fontsize=14)
    ax.set_title('Enhanced THz ISL ISAC C-D Trade-off\n' + 
                 f'(1m Antennas, 30 dBm Tx Power, 2000 km Distance)',
                 fontsize=16, pad=15)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    
    ax.set_xscale('log')
    ax.set_xlim(left=0.1)
    ax.set_ylim(bottom=0)
    
    # Add performance regions
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5)
    ax.text(0.5, 2.1, 'Good communication (>2 bits/symbol)', 
            transform=ax.get_yaxis_transform(), 
            fontsize=10, color='green', ha='center')
    
    ax.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5)
    ax.text(0.9, 0.95, 'Sub-mm\naccuracy', 
            transform=ax.get_xaxis_transform(),
            fontsize=10, color='blue', ha='right', va='top')
    
    plt.tight_layout()
    plt.savefig('cd_frontier_enhanced.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cd_frontier_enhanced.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function with all analyses."""
    print("=== Enhanced THz ISL ISAC Analysis Suite ===")
    print("\nKey Improvements:")
    print("- Larger antennas (1m) for better link budget")
    print("- Higher transmit power (30 dBm)")
    print("- Multiple analysis perspectives")
    
    # Set debug level
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = False
    
    # 1. SNR to hardware limit analysis
    plot_snr_to_hardware_limit()
    
    # 2. Hardware quality factor sensitivity
    plot_gamma_eff_sensitivity()
    
    # 3. 3D performance landscape
    plot_3d_performance_landscape()
    
    # 4. ISAC feasibility regions
    plot_isac_feasibility_regions()
    
    # 5. Enhanced C-D frontier
    plot_cd_frontier_enhanced()
    
    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("1. SNR to Hardware Limit - Shows transition from power-limited to hardware-limited")
    print("2. Hardware Sensitivity - Quantifies performance vs Gamma_eff")
    print("3. 3D Performance Landscape - Optimal frequency/distance selection")
    print("4. ISAC Feasibility Regions - Design space exploration")
    print("5. Enhanced C-D Frontier - Complete trade-off analysis")

if __name__ == "__main__":
    main()