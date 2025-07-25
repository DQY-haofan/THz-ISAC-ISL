#!/usr/bin/env python3
"""
cd_frontier_simulation.py - FIXED VERSION

关键修正：
1. 只考虑可观测参数（距离、径向速度）
2. 修正噪声计算
3. 修正FIM维度
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

# Use a professional color palette
colors = sns.color_palette("husl", 4)

class ISACSystem:
    """Class to encapsulate ISAC system parameters and methods."""
    
    def __init__(self, hardware_profile: str, f_c: float = 300e9, 
                 distance: float = 2000e3, n_pilots: int = 64):
        """
        Initialize ISAC system with given parameters.
        
        Args:
            hardware_profile: Name of hardware profile ('High_Performance' or 'SWaP_Efficient')
            f_c: Carrier frequency [Hz]
            distance: ISL distance [m]
            n_pilots: Number of pilot symbols
        """
        self.profile = HARDWARE_PROFILES[hardware_profile]
        self.f_c = f_c
        self.distance = distance
        self.n_pilots = n_pilots
        
        # Calculate system parameters
        self.lambda_c = PhysicalConstants.c / f_c
        self.channel_gain = self._calculate_channel_gain()
        self.bussgang_gain = self._calculate_bussgang_gain()
        
        # 添加：计算路径损耗以便调试
        self.path_loss_dB = 20 * np.log10(4 * np.pi * self.distance / self.lambda_c)
        print(f"  Path loss at {f_c/1e9:.0f} GHz, {distance/1e3:.0f} km: {self.path_loss_dB:.1f} dB")
        
        # Constellation design
        self.constellation = self._create_constellation()
    
    def _calculate_channel_gain(self) -> float:
        """Calculate channel gain magnitude |g|."""
        antenna_gain = scenario.antenna_gain ** 2  # G_tx * G_rx
        beta_ch = (self.lambda_c / (4 * np.pi * self.distance)) * np.sqrt(antenna_gain)
        return beta_ch
    
    def _calculate_bussgang_gain(self) -> float:
        """Calculate Bussgang gain |B| for PA."""
        # Assume 7 dB input backoff
        kappa = 10 ** (-7.0 / 10)
        B = 1 - 1.5 * kappa + 1.875 * kappa**2
        return B
    
    def _create_constellation(self, modulation: str = 'QPSK') -> np.ndarray:
        """Create normalized constellation points."""
        if modulation == 'QPSK':
            # QPSK constellation with unit average power
            angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            constellation = np.exp(1j * angles)
            # Normalize to unit average power
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        elif modulation == '16QAM':
            # 16-QAM constellation
            levels = [-3, -1, 1, 3]
            constellation = []
            for i in levels:
                for q in levels:
                    constellation.append(i + 1j*q)
            constellation = np.array(constellation)
            # Normalize to unit average power
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        else:
            raise ValueError(f"Unsupported modulation: {modulation}")
        
        return constellation
    
    def calculate_sinr(self, symbol: complex, avg_power: float, snr_nominal: float) -> float:
        """
        Calculate SINR for a given symbol.
        
        Args:
            symbol: Complex constellation point
            avg_power: Average transmit power
            snr_nominal: Nominal SNR (linear)
            
        Returns:
            SINR value (linear)
        """
        # Symbol power
        p_symbol = np.abs(symbol)**2 * avg_power
        
        # 接收功率
        p_rx = p_symbol * self.channel_gain**2 * self.bussgang_gain**2
        
        # 热噪声（基于发射端SNR定义）
        N_0 = avg_power / snr_nominal
        
        # 硬件相关噪声
        N_hw = p_rx * self.profile.Gamma_eff
        
        # 相位噪声影响
        phase_factor = np.exp(self.profile.phase_noise_variance)
        
        # 总噪声
        noise_total = N_0 + N_hw * phase_factor
        
        # SINR
        sinr = p_rx / noise_total
        
        return sinr
    
    def calculate_mutual_information(self, p_x: np.ndarray, snr_nominal: float) -> np.ndarray:
        """
        Calculate mutual information for each symbol.
        
        Args:
            p_x: Probability distribution over constellation
            snr_nominal: Nominal SNR (linear)
            
        Returns:
            Array of I(x) values for each symbol
        """
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        I_x = np.zeros(len(self.constellation))
        
        for i, symbol in enumerate(self.constellation):
            sinr = self.calculate_sinr(symbol, avg_power, snr_nominal)
            I_x[i] = np.log2(1 + sinr)  # 复基带信道，无1/2因子
            
        return I_x
    
    def calculate_bfim_observable(self, avg_power: float, snr_nominal: float) -> np.ndarray:
        """
        Calculate Bayesian Fisher Information Matrix for OBSERVABLE parameters.
        
        关键修正：只考虑单链路可观测的参数
        - 距离 R
        - 径向速度 v_r
        
        Args:
            avg_power: Average transmit power
            snr_nominal: Nominal SNR (linear)
            
        Returns:
            2x2 B-FIM for observable parameters
        """
        # 接收功率
        P_tx = avg_power
        P_rx = P_tx * self.channel_gain**2 * self.bussgang_gain**2
        
        # 计算有效噪声
        N_0 = P_tx / snr_nominal
        N_hw = P_rx * self.profile.Gamma_eff
        phase_factor = np.exp(self.profile.phase_noise_variance)
        sigma_eff_sq = N_0 + N_hw * phase_factor
        
        # FIM缩放因子
        fim_scale = (2 * self.n_pilots * P_rx * np.exp(-self.profile.phase_noise_variance)) / sigma_eff_sq
        
        # 距离测量的FIM（载波相位灵敏度）
        phase_sensitivity = (2 * np.pi * self.f_c / PhysicalConstants.c)**2
        J_range = fim_scale * phase_sensitivity
        
        # 径向速度测量的FIM（多普勒灵敏度）
        # 假设观测时间T
        T_obs = self.n_pilots * 1e-6  # 假设每个符号1μs
        doppler_sensitivity = (2 * np.pi * self.f_c * T_obs / PhysicalConstants.c)**2
        J_velocity = fim_scale * doppler_sensitivity
        
        # 构建2x2 FIM矩阵（对角阵，参数独立）
        J_B = np.diag([J_range, J_velocity])
        
        return J_B
    
    def calculate_distortion(self, p_x: np.ndarray, snr_nominal: float) -> float:
        """
        Calculate sensing distortion as Tr(J_B^{-1}) for observable parameters.
        修正版：确保功率变化正确传播到 FIM
        """
        avg_power = np.sum(p_x * np.abs(self.constellation)**2)
        
        # 防止功率过小
        if avg_power < 1e-10:
            return 1e10
        
        # 添加调试输出
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
        
        if self._debug_count % 10 == 0:  # 每10次迭代打印一次
            print(f"\nDebug distortion calculation:")
            print(f"  avg_power = {avg_power:.6f}")
            print(f"  p_x entropy = {-np.sum(p_x * np.log2(p_x + 1e-10)):.3f}")
        
        J_B = self.calculate_bfim_observable(avg_power, snr_nominal)
        
        try:
            J_B_inv = inv(J_B)
            distortion = np.trace(J_B_inv)
            
            # 添加功率依赖性检查
            if self._debug_count % 10 == 0:
                print(f"  distortion = {distortion:.6e}")
                print(f"  J_B condition number = {np.linalg.cond(J_B):.2e}")
            
            return distortion
            
        except np.linalg.LinAlgError:
            print("    Warning: FIM singular")
            return 1e10

def modified_blahut_arimoto(system: ISACSystem, D_target: float, 
                           snr_nominal: float = 100,  # 20 dB
                           epsilon_lambda: float = 1e-3,
                           epsilon_p: float = 1e-6,
                           max_iterations: int = 50,
                           verbose: bool = False) -> Tuple[float, np.ndarray]:
    """
    Modified Blahut-Arimoto algorithm for ISAC C-D trade-off.
    
    修正版：使用可观测参数的失真度量
    """
    n_symbols = len(system.constellation)
    
    # Initialize uniform distribution
    p_x = np.ones(n_symbols) / n_symbols
    
    # 记录收敛历史
    convergence_history = {
        'lambda': [],
        'distortion': [],
        'capacity': []
    }

    # Binary search bounds for Lagrange multiplier
    lambda_min = 0
    lambda_max = 1000  # Initial upper bound
    
    # Find initial lambda_max that gives distortion > D_target
    p_uniform = np.ones(n_symbols) / n_symbols
    D_uniform = system.calculate_distortion(p_uniform, snr_nominal)
    if D_uniform <= D_target:
        # If uniform distribution already meets constraint, no trade-off needed
        I_x = system.calculate_mutual_information(p_uniform, snr_nominal)
        return np.sum(p_uniform * I_x), p_uniform
    
    # Binary search for optimal lambda
    while (lambda_max - lambda_min) > epsilon_lambda:
        lambda_current = (lambda_min + lambda_max) / 2
        
        if verbose:
            print(f"  Lambda: {lambda_current:.4f}")
        
        # Reset distribution
        p_x = np.ones(n_symbols) / n_symbols
        
        # Inner loop: optimize p_x for fixed lambda
        for inner_iter in range(max_iterations):
            p_x_prev = p_x.copy()
            
            # Step 1: Compute per-symbol mutual information
            I_x = system.calculate_mutual_information(p_x, snr_nominal)
            
            # Step 2: Compute distortion and its gradient
            D_current = system.calculate_distortion(p_x, snr_nominal)
            
            # Approximate distortion gradient (numerical)
            grad_D = np.zeros(n_symbols)
            delta = 1e-6
            for i in range(n_symbols):
                p_perturb = p_x.copy()
                p_perturb[i] += delta
                p_perturb /= p_perturb.sum()  # Renormalize
                D_perturb = system.calculate_distortion(p_perturb, snr_nominal)
                grad_D[i] = (D_perturb - D_current) / delta
            
            # Step 3: Update distribution via exponentiated gradient
            beta = I_x - lambda_current * grad_D
            beta_max = np.max(beta)  # For numerical stability
            p_x = p_x * np.exp(beta - beta_max)
            p_x /= p_x.sum()  # Normalize
            
            # Check convergence
            if np.linalg.norm(p_x - p_x_prev, ord=1) < epsilon_p:
                break
        
        # Step 4: Update lambda bounds
        if D_current > D_target:
            lambda_min = lambda_current  # Need higher penalty
        else:
            lambda_max = lambda_current  # Can reduce penalty

                # 在内循环中添加：
        if verbose and inner_iter % 10 == 0:
            print(f"    Inner iter {inner_iter}: ||Δp||₁ = {np.linalg.norm(p_x - p_x_prev, ord=1):.6f}")
            print(f"    Current distribution entropy: {-np.sum(p_x * np.log2(p_x + 1e-10)):.3f}")
                
    # Calculate final capacity
    I_x_final = system.calculate_mutual_information(p_x, snr_nominal)
    capacity = np.sum(p_x * I_x_final)
    
    if verbose:
        print(f"  Final distortion: {D_current:.6f}, Capacity: {capacity:.4f}")
    
    return capacity, p_x

def generate_cd_frontier(hardware_profile: str, 
                        n_distortion_points: int = 20,
                        snr_dB: float = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Capacity-Distortion frontier for a given hardware profile.
    
    修正版：使用可观测参数
    """
    print(f"\nGenerating C-D frontier for {hardware_profile}...")
    
    # Create system
    system = ISACSystem(hardware_profile)
    snr_linear = 10 ** (snr_dB / 10)
    
    # Get distortion range
    # Maximum distortion (uniform distribution, low power)
    p_uniform = np.ones(len(system.constellation)) / len(system.constellation)
    D_max = system.calculate_distortion(p_uniform * 0.1, snr_linear)  # Low power
    D_min = system.calculate_distortion(p_uniform, snr_linear) / 100  # High power, optimistic
    
    # Log-spaced distortion targets
    D_targets = np.logspace(np.log10(D_min), np.log10(D_max), n_distortion_points)
    
    capacities = []
    distortions = []
    
    # Use progress bar
    for D_target in tqdm(D_targets, desc=f"{hardware_profile}"):
        try:
            capacity, p_opt = modified_blahut_arimoto(system, D_target, snr_linear)
            
            # Verify actual distortion
            actual_D = system.calculate_distortion(p_opt, snr_linear)
            
            capacities.append(capacity)
            distortions.append(actual_D)
        except:
            # Skip problematic points
            continue
    
    return np.array(distortions), np.array(capacities)

def plot_cd_frontier():
    """Generate and plot the Capacity-Distortion frontier figure."""
    print("=== Generating Capacity-Distortion Frontier Plot ===")
    
    # Operating conditions
    snr_dB = 20  # Moderate SNR to see trade-off
    
    # Generate frontiers for both profiles
    profiles = ["High_Performance", "SWaP_Efficient"]
    results = {}
    
    for profile in profiles:
        distortions, capacities = generate_cd_frontier(profile, n_distortion_points=15, snr_dB=snr_dB)
        results[profile] = {
            'distortions': distortions,
            'capacities': capacities,
            'ranging_rmse': np.sqrt(distortions)  # Convert to ranging RMSE
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot frontiers
    for i, profile in enumerate(profiles):
        data = results[profile]
        
        if len(data['ranging_rmse']) > 0:  # Check if we have data
            # Convert distortion to ranging precision
            # 使用距离RMSE的倒数作为精度度量
            ranging_rmse_m = data['ranging_rmse']
            
            # Plot frontier
            ax.plot(ranging_rmse_m * 1000, data['capacities'],  # Convert to mm
                    color=colors[i], linewidth=3,
                    marker='o', markersize=8,
                    label=profile.replace('_', ' '),
                    markerfacecolor='white',
                    markeredgewidth=2)
            
            # Add shaded region under curve
            ax.fill_between(ranging_rmse_m * 1000, 0, data['capacities'], 
                           alpha=0.2, color=colors[i])
    
    # Formatting
    ax.set_xlabel('Ranging RMSE [mm]', fontsize=14)
    ax.set_ylabel('Communication Capacity [bits/symbol]', fontsize=14)
    ax.set_title(f'ISAC Capacity-Distortion Trade-off Frontier\n' + 
                f'(Observable Parameters: Range & Radial Velocity)\n' +
                f'SNR = {snr_dB} dB, f_c = 300 GHz, Distance = 2000 km',
                fontsize=16, pad=15)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, shadow=True)
    
    # Set reasonable axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Use log scale for x-axis if range is large
    ax.set_xscale('log')
    
    # Add annotations
    ax.annotate('Better sensing,\nlower capacity', 
                xy=(0.15, 0.3), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    ax.annotate('Better communication,\nworse sensing', 
                xy=(0.85, 0.7), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    # Save figure
    plt.tight_layout()
    plt.savefig('cd_frontier.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cd_frontier.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical results
    print("\n=== Capacity-Distortion Trade-off Results ===")
    print("(Based on observable parameters: range and radial velocity)")
    for profile in profiles:
        data = results[profile]
        if len(data['ranging_rmse']) > 0:
            print(f"\n{profile}:")
            print(f"  Ranging RMSE range: {data['ranging_rmse'].min()*1000:.3f} - {data['ranging_rmse'].max()*1000:.3f} mm")
            print(f"  Capacity range: {data['capacities'].min():.3f} - {data['capacities'].max():.3f} bits/symbol")
            
            # Find balanced point
            mid_idx = len(data['capacities']) // 2
            if mid_idx < len(data['capacities']):
                print(f"  Balanced point: {data['ranging_rmse'][mid_idx]*1000:.3f} mm, "
                      f"{data['capacities'][mid_idx]:.3f} bits/symbol")

def plot_constellation_evolution():
    """Additional plot showing how constellation distribution changes along frontier."""
    print("\n=== Generating Constellation Distribution Evolution ===")
    
    # Create system
    system = ISACSystem("SWaP_Efficient")
    snr_linear = 100  # 20 dB
    
    # Three points on frontier: sensing-focused, balanced, comm-focused
    D_targets = [1e-8, 1e-6, 1e-4]  # Different distortion targets for observable params
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (D_target, ax) in enumerate(zip(D_targets, axes)):
        try:
            capacity, p_x = modified_blahut_arimoto(system, D_target, snr_linear, verbose=False)
            
            # Plot constellation with probabilities as sizes
            for j, (symbol, prob) in enumerate(zip(system.constellation, p_x)):
                ax.scatter(symbol.real, symbol.imag, 
                          s=prob*2000,  # Scale for visibility
                          color=colors[0], alpha=0.7,
                          edgecolors='black', linewidth=1.5)
                ax.text(symbol.real, symbol.imag + 0.15, f'{prob:.2f}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('In-phase', fontsize=11)
            ax.set_ylabel('Quadrature', fontsize=11)
            
            # Calculate actual distortion
            actual_D = system.calculate_distortion(p_x, snr_linear)
            ranging_rmse_mm = np.sqrt(actual_D) * 1000
            
            if i == 0:
                ax.set_title(f'Sensing-Focused\n(RMSE = {ranging_rmse_mm:.1f} mm, C = {capacity:.2f})', fontsize=12)
            elif i == 1:
                ax.set_title(f'Balanced\n(RMSE = {ranging_rmse_mm:.1f} mm, C = {capacity:.2f})', fontsize=12)
            else:
                ax.set_title(f'Comm-Focused\n(RMSE = {ranging_rmse_mm:.1f} mm, C = {capacity:.2f})', fontsize=12)
        except:
            ax.set_title(f'Failed to converge', fontsize=12)
    
    plt.suptitle('Constellation Probability Distribution Evolution Along C-D Frontier\n(Observable Parameters Only)', fontsize=14)
    plt.tight_layout()
    plt.savefig('constellation_evolution.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('constellation_evolution.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run C-D frontier simulations."""
    print("=== THz ISL ISAC Capacity-Distortion Frontier Simulation ===")
    print("Implementing Modified Blahut-Arimoto Algorithm")
    print("\nIMPORTANT: Using only observable parameters for single ISL:")
    print("  - Range (distance)")
    print("  - Radial velocity")
    print("  (Full 3D position requires multiple non-coplanar links)")
    
    # Generate main C-D frontier plot
    plot_cd_frontier()
    
    # Generate constellation evolution plot
    plot_constellation_evolution()
    
    print("\n=== Simulation Complete ===")
    print("Output files:")
    print("  - cd_frontier.pdf/png")
    print("  - constellation_evolution.pdf/png")

if __name__ == "__main__":
    main()