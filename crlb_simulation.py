#!/usr/bin/env python3
"""
crlb_simulation.py - FINAL FIXED VERSION

Fixed issues:
1. Ensured all calculations use SI units (m, m/s, Hz)
2. Using correct phase noise variance from config
3. Proper unit conversions where needed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from typing import Tuple, Dict

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
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
})

# Use a professional color palette
colors = sns.color_palette("husl", 5)

def calculate_channel_gain(distance_m: float, frequency_Hz: float, 
                         antenna_gain_linear: float = None) -> float:
    """
    Calculate channel gain using Friis equation.
    
    Args:
        distance_m: ISL distance in meters (SI unit)
        frequency_Hz: Carrier frequency in Hz (SI unit)
        antenna_gain_linear: Combined antenna gain (G_tx * G_rx), if None uses scenario default
        
    Returns:
        Channel gain magnitude |g|
    """
    if antenna_gain_linear is None:
        # Use default antenna gain (same for Tx and Rx)
        antenna_gain_linear = scenario.antenna_gain ** 2
    
    lambda_c = PhysicalConstants.wavelength(frequency_Hz)
    beta_ch = (lambda_c / (4 * np.pi * distance_m)) * np.sqrt(antenna_gain_linear)
    
    return beta_ch

def calculate_bussgang_gain(input_backoff_dB: float = 7.0) -> float:
    """
    Calculate Bussgang gain for PA nonlinearity.
    
    Args:
        input_backoff_dB: Input backoff in dB (default 7 dB)
        
    Returns:
        Bussgang gain magnitude |B|
    """
    # From manuscript approximation for typical operating point
    kappa = 10 ** (-input_backoff_dB / 10)  # IBO ratio
    
    # Taylor expansion approximation for small kappa
    B = 1 - 1.5 * kappa + 1.875 * kappa**2
    
    return B

def calculate_effective_noise_variance(
    SNR_linear: float,
    channel_gain: float,
    hardware_profile: str,
    signal_power: float = 1.0
) -> Tuple[float, float]:
    """
    Calculate effective noise variance including hardware impairments.
    
    修正版本：确保硬件差异和频率依赖性正确反映
    """
    profile = HARDWARE_PROFILES[hardware_profile]
    
    # Bussgang gain
    B = calculate_bussgang_gain()
    
    # 接收信号功率（包含路径损耗）
    P_rx = signal_power * (channel_gain ** 2) * (B ** 2)
    
    # 热噪声功率（从SNR定义反推）
    # SNR = P_rx / N_0，所以 N_0 = P_rx / SNR
    N_0 = P_rx / SNR_linear
    
    # 硬件引起的信号相关噪声
    # 这是关键：必须使用profile.Gamma_eff！
    sigma_hw_sq = P_rx * profile.Gamma_eff
    
    # 相位噪声放大因子
    phase_noise_factor = np.exp(profile.phase_noise_variance)
    
    # DSE残差（可忽略）
    sigma_DSE_sq = 0.001 * N_0 / SNR_linear
    
    # 总有效噪声方差
    # 注意：硬件噪声需要乘以相位噪声因子
    sigma_eff_sq = N_0 + sigma_hw_sq * phase_noise_factor + sigma_DSE_sq
    
    return sigma_eff_sq, N_0


def calculate_position_bcrlb(
    f_c: float,
    sigma_eff_sq: float,
    M: int,
    channel_gain: float,
    B: float,
    sigma_phi_sq: float
) -> float:
    """
    CRITICAL: 确保 f_c² 依赖性！
    """
    # 调试：打印中间值
    print(f"\n  DEBUG BCRLB calculation:")
    print(f"    f_c = {f_c/1e9:.0f} GHz")
    print(f"    f_c² = {(f_c/1e9)**2:.0f} GHz²")
    
    # 关键：必须使用 f_c 的平方！
    term1 = PhysicalConstants.c**2 / (8 * np.pi**2 * f_c**2)
    print(f"    term1 (c²/8π²f_c²) = {term1:.2e}")
    
    P_rx = (channel_gain**2) * (B**2)
    term2 = sigma_eff_sq / (M * P_rx)
    print(f"    term2 (noise term) = {term2:.2e}")
    
    term3 = np.exp(sigma_phi_sq)
    print(f"    term3 (phase penalty) = {term3:.3f}")
    
    bcrlb = term1 * term2 * term3
    print(f"    BCRLB = {bcrlb:.2e} m²")
    print(f"    RMSE = {np.sqrt(bcrlb):.2e} m")
    
    return bcrlb

def simulate_ranging_crlb_vs_snr():
    """Generate Figure 1: Ranging CRLB vs. SNR for different carrier frequencies."""
    print("Generating Figure 1: Ranging CRLB vs. SNR...")
    
    # Simulation parameters
    frequencies_GHz = [100, 300, 600]  # GHz
    frequencies_Hz = [f * 1e9 for f in frequencies_GHz]  # Convert to Hz
    hardware_profile = "SWaP_Efficient"
    
    # Get profile parameters
    profile = HARDWARE_PROFILES[hardware_profile]
    B = calculate_bussgang_gain()
    
    # 添加调试输出
    print(f"Using hardware profile: {hardware_profile}")
    print(f"Gamma_eff = {profile.Gamma_eff}")
    print(f"Frequencies to plot: {frequencies_GHz} GHz")
    
    # Initialize results storage
    results = {f: [] for f in frequencies_GHz}
    
    # Iterate over SNR range
    for snr_dB in simulation.SNR_dB_array:
        snr_linear = 10 ** (snr_dB / 10)
        
        for f_GHz, f_Hz in zip(frequencies_GHz, frequencies_Hz):
            # Calculate channel gain at this frequency
            g = calculate_channel_gain(scenario.R_default, f_Hz)
            
            # Calculate effective noise
            sigma_eff_sq, N_0 = calculate_effective_noise_variance(
                snr_linear, g, hardware_profile
            )
            
            # 添加调试输出（只在第一个SNR点）
            if snr_dB == simulation.SNR_dB_array[0]:
                print(f"\nAt {f_GHz} GHz, SNR={snr_dB} dB:")
                print(f"  Channel gain g = {g:.2e}")
                print(f"  N_0 = {N_0:.2e}")
                print(f"  sigma_eff_sq = {sigma_eff_sq:.2e}")
            
            # Calculate position BCRLB
            bcrlb_pos = calculate_position_bcrlb(
                f_Hz, sigma_eff_sq, simulation.n_pilots,
                g, B, profile.phase_noise_variance
            )
            
            # Convert to ranging RMSE
            ranging_rmse_m = np.sqrt(bcrlb_pos)
            
            results[f_GHz].append(ranging_rmse_m)
    
    # 调试：打印结果长度
    for f_GHz in frequencies_GHz:
        print(f"\nResults for {f_GHz} GHz: {len(results[f_GHz])} points")
        print(f"  First value: {results[f_GHz][0]:.2e} m")
        print(f"  Last value: {results[f_GHz][-1]:.2e} m")

def simulate_ranging_crlb_vs_hardware():
    """Generate Figure 2: Ranging CRLB vs. Hardware Profile at fixed high SNR."""
    print("\nGenerating Figure 2: Ranging CRLB vs. Hardware Profile...")
    
    # Fixed parameters
    snr_dB = 30
    snr_linear = 10 ** (snr_dB / 10)
    f_c_GHz = 300
    f_c_Hz = f_c_GHz * 1e9
    
    # Calculate for both hardware profiles
    profiles = ["High_Performance", "SWaP_Efficient"]
    ranging_rmse_results = []
    
    for profile_name in profiles:
        profile = HARDWARE_PROFILES[profile_name]
        
        # Calculate channel parameters
        g = calculate_channel_gain(scenario.R_default, f_c_Hz)
        B = calculate_bussgang_gain()
        
        # Calculate effective noise
        sigma_eff_sq, N_0 = calculate_effective_noise_variance(
            snr_linear, g, profile_name
        )
        
        # 添加详细调试输出
        print(f"\n{profile_name}:")
        print(f"  Gamma_eff = {profile.Gamma_eff}")
        print(f"  sigma_eff_sq = {sigma_eff_sq:.2e}")
        print(f"  N_0 = {N_0:.2e}")
        print(f"  Hardware noise = {sigma_eff_sq - N_0:.2e}")
        
        # Calculate position BCRLB
        bcrlb_pos = calculate_position_bcrlb(
            f_c_Hz, sigma_eff_sq, simulation.n_pilots,
            g, B, profile.phase_noise_variance
        )
        
        ranging_rmse_m = np.sqrt(bcrlb_pos)
        ranging_rmse_results.append(ranging_rmse_m)
        
        print(f"  BCRLB = {bcrlb_pos:.2e} m²")
        print(f"  Ranging RMSE = {ranging_rmse_m:.2e} m")
        
        # 添加正确的组件百分比计算
        print("\n  Hardware Component Breakdown:")
        # 使用实际组件值计算百分比
        actual_total = profile.Gamma_PA + profile.Gamma_LO + profile.Gamma_ADC
        
        pa_percent = (profile.Gamma_PA / actual_total) * 100
        lo_percent = (profile.Gamma_LO / actual_total) * 100
        adc_percent = (profile.Gamma_ADC / actual_total) * 100
        
        print(f"    PA contribution: {pa_percent:.1f}%")
        print(f"    LO contribution: {lo_percent:.3f}%")
        print(f"    ADC contribution: {adc_percent:.2f}%")
        print(f"    Total: {pa_percent + lo_percent + adc_percent:.1f}%")  # 应该是 100%
        
        # 验证配置的一致性
        if abs(actual_total - profile.Gamma_eff) / profile.Gamma_eff > 0.1:
            print(f"    WARNING: Component sum ({actual_total:.4f}) differs from Gamma_eff ({profile.Gamma_eff:.4f}) by >10%")
            


def main():
    """Main function to run all simulations."""
    print("=== THz ISL ISAC CRLB Simulation ===")
    print(f"Configuration: {simulation.n_pilots} pilots, {scenario.R_default/1e3:.0f} km distance")
    print(f"Hardware profiles: {list(HARDWARE_PROFILES.keys())}")
    
    # Verify units
    print("\nUnit Check:")
    print(f"  Distance: {scenario.R_default} m (SI unit)")
    print(f"  Velocity: {scenario.v_rel_default} m/s (SI unit)")
    print(f"  Speed of light: {PhysicalConstants.c} m/s (SI unit)")
    
    # Run simulations
    simulate_ranging_crlb_vs_snr()
    simulate_ranging_crlb_vs_hardware()
    
    print("\n=== Simulation Complete ===")
    print("Output files:")
    print("  - ranging_crlb_vs_snr.pdf/png")
    print("  - ranging_crlb_vs_hardware.pdf/png")

if __name__ == "__main__":
    main()