#!/usr/bin/env python3
"""
simulation_config.py - UPDATED VERSION WITH CORRECT PARAMETERS

Key fixes:
1. Adjusted SWaP_Efficient Gamma_eff to 0.025 (from supporting document)
2. Added option for larger antenna diameter for better link budget
3. Verified phase noise variance calculation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# =============================================================================
# PHYSICAL CONSTANTS (SI UNITS)
# =============================================================================
class PhysicalConstants:
    """Fundamental physical constants in SI units."""
    
    c = 3e8                      # Speed of light [m/s]
    k = 1.380649e-23            # Boltzmann's constant [J/K]
    T_noise = 290               # Standard noise temperature [K]
    
    @classmethod
    def wavelength(cls, frequency_hz):
        """Calculate wavelength from frequency."""
        return cls.c / frequency_hz

# =============================================================================
# SCENARIO PARAMETERS (SI UNITS)
# =============================================================================
@dataclass
class ScenarioParameters:
    """Parameters defining the LEO ISL scenario - ALL IN SI UNITS."""
    
    # Carrier frequency range [Hz]
    f_c_min: float = 100e9      # Minimum carrier frequency [Hz] (100 GHz)
    f_c_max: float = 600e9      # Maximum carrier frequency [Hz] (600 GHz)
    f_c_default: float = 300e9  # Default carrier frequency [Hz] (300 GHz)
    
    # ISL geometry [meters]
    R_min: float = 500e3        # Minimum ISL distance [m] (500 km)
    R_max: float = 5000e3       # Maximum ISL distance [m] (5000 km)
    R_default: float = 2000e3   # Default ISL distance [m] (2000 km)
    
    # Relative dynamics [m/s]
    v_rel_max: float = 15e3     # Maximum relative velocity [m/s] (15 km/s)
    v_rel_default: float = 10e3 # Default relative velocity [m/s] (10 km/s)
    a_rel_max: float = 100      # Maximum relative acceleration [m/s²]
    
    # Antenna parameters
    D_antenna: float = 0.5      # Default antenna diameter [m]
    D_antenna_large: float = 1.0  # Large antenna option [m]
    eta_antenna: float = 0.55   # Antenna efficiency
    
    # Transmit power options
    P_tx_dBm_default: float = 20  # Default: 20 dBm (100 mW)
    P_tx_dBm_high: float = 30     # High power: 30 dBm (1 W)
    
    # Derived antenna parameters
    def antenna_gain(self, diameter: float = None, frequency_hz: float = None) -> float:
        """Calculate antenna gain [linear] for given diameter and frequency."""
        if diameter is None:
            diameter = self.D_antenna
        if frequency_hz is None:
            frequency_hz = self.f_c_default
            
        lambda_c = PhysicalConstants.wavelength(frequency_hz)
        return self.eta_antenna * (np.pi * diameter / lambda_c) ** 2
    
    def antenna_gain_dB(self, diameter: float = None, frequency_hz: float = None) -> float:
        """Antenna gain in dB."""
        return 10 * np.log10(self.antenna_gain(diameter, frequency_hz))
    
    def beamwidth_3dB(self, frequency_hz: float, diameter: float = None) -> float:
        """Calculate 3dB beamwidth [rad] at given frequency."""
        if diameter is None:
            diameter = self.D_antenna
        lambda_c = PhysicalConstants.wavelength(frequency_hz)
        return 1.02 * lambda_c / diameter
    
    def beam_rolloff_factor(self, frequency_hz: float, diameter: float = None) -> float:
        """Calculate beam rolloff factor γ = 2.77/θ_3dB² [rad⁻²]."""
        theta_3dB = self.beamwidth_3dB(frequency_hz, diameter)
        return 2.77 / (theta_3dB ** 2)

# =============================================================================
# HARDWARE PROFILES - CORRECTED VALUES
# =============================================================================
@dataclass
class HardwareComponentSpecs:
    """Specifications for individual hardware components."""
    
    # Power Amplifier
    PA_technology: str          # Technology type (e.g., "InP DHBT", "CMOS with DPD")
    PA_EVM_percent: float       # PA Error Vector Magnitude [%]
    PA_P_sat_dBm: float        # PA saturation power [dBm]
    PA_efficiency: float        # Power-added efficiency
    
    # Local Oscillator / PLL
    LO_technology: str          # Technology type (e.g., "28nm CMOS", "SiGe BiCMOS")
    LO_RMS_jitter_fs: float    # RMS timing jitter [femtoseconds]
    LO_linewidth_Hz: float     # 3dB linewidth [Hz]
    
    # ADC
    ADC_technology: str         # Technology type
    ADC_ENOB: float            # Effective Number of Bits
    ADC_sampling_rate_Gsps: float  # Sampling rate [Gsamples/s]

@dataclass
class HardwareProfile:
    """Complete hardware profile with quality factors."""
    
    name: str
    description: str
    
    # Aggregate hardware quality factor
    Gamma_eff: float           # Total hardware quality factor (EVM²)
    
    # Component-level contributions
    Gamma_PA: float            # PA contribution to Gamma_eff
    Gamma_LO: float            # LO contribution to Gamma_eff  
    Gamma_ADC: float           # ADC contribution to Gamma_eff
    
    # Component specifications
    components: HardwareComponentSpecs
    
    # System parameters
    frame_duration_s: float    # Frame duration [seconds]
    signal_bandwidth_Hz: float # Signal bandwidth [Hz]
    
    @property
    def phase_noise_variance(self) -> float:
        """
        Calculate phase noise variance σ_φ² [rad²].
        Using paper's formula: σ_φ² ≈ (4/3) * π * Δν * T
        """
        delta_nu = self.components.LO_linewidth_Hz
        T = self.frame_duration_s
        
        # From paper's approximation for Wiener process
        variance = (4/3) * np.pi * delta_nu * T
        
        # Print warning if variance is too large
        if variance > 0.1:
            print(f"WARNING: Phase noise variance {variance:.3f} rad² may be too large for approximations!")
        
        return variance
    
    @property
    def coherence_time(self) -> float:
        """
        Estimate coherence time for σ_φ² ≈ 0.1 rad² (reasonable limit).
        """
        delta_nu = self.components.LO_linewidth_Hz
        # Solve: (4/3) * π * Δν * T_coh = 0.1
        return 0.1 / ((4/3) * np.pi * delta_nu)
    
    @property
    def EVM_total_percent(self) -> float:
        """Total system EVM [%]."""
        return 100 * np.sqrt(self.Gamma_eff)

# Define hardware profiles based on the supporting document
HIGH_PERFORMANCE_PROFILE = HardwareProfile(
    name="High_Performance",
    description="III-V semiconductor based system optimized for performance",
    
    # Aggregate quality factor (from supporting document)
    Gamma_eff=0.01,
    
    # Component contributions (from Table 5 synthesis)
    Gamma_PA=0.0112,          # InP DHBT PA with -19.5 dB EVM
    Gamma_LO=4.3e-7,          # Based on 20.9 fs RMS jitter
    Gamma_ADC=1.7e-4,         # Based on ~6.0 ENOB
    
    # Component specifications
    components=HardwareComponentSpecs(
        # PA specs (220 GHz InP DHBT)
        PA_technology="InP DHBT",
        PA_EVM_percent=10.6,   # -19.5 dB EVM
        PA_P_sat_dBm=15,       # Typical for InP at sub-THz
        PA_efficiency=0.15,     # ~15% PAE for InP
        
        # LO specs (28nm CMOS PLL)
        LO_technology="28nm CMOS",
        LO_RMS_jitter_fs=20.9,
        LO_linewidth_Hz=100e3,  # 100 kHz
        
        # ADC specs (20nm CMOS)
        ADC_technology="20nm CMOS",
        ADC_ENOB=5.95,
        ADC_sampling_rate_Gsps=20
    ),
    
    # Frame duration for σ_φ² ≈ 0.042 (from paper)
    frame_duration_s=0.1e-6,    # 0.1 μs = 100 ns
    signal_bandwidth_Hz=10e9    # 10 GHz
)

# CORRECTED: Using 0.025 from supporting document instead of 0.045
SWAP_EFFICIENT_PROFILE = HardwareProfile(
    name="SWaP_Efficient", 
    description="Silicon-based system optimized for SWaP and cost with DPD",
    
    # CORRECTED: Aggregate quality factor from supporting document
    Gamma_eff=0.025,  # Changed from 0.045 to match supporting document
    
    # Component contributions - adjusted proportionally
    Gamma_PA=0.0222,          # SiGe with DPD (from supporting doc)
    Gamma_LO=4.8e-6,          # Based on 70 fs RMS jitter
    Gamma_ADC=6.5e-4,         # Based on 5.0 ENOB
    
    # Component specifications
    components=HardwareComponentSpecs(
        # PA specs (SiGe with DPD)
        PA_technology="SiGe BiCMOS with DPD",
        PA_EVM_percent=14.9,    # -16.5 dB EVM (from supporting doc)
        PA_P_sat_dBm=10,        # Lower than InP
        PA_efficiency=0.05,      # ~5% PAE for CMOS at THz
        
        # LO specs (SiGe BiCMOS)
        LO_technology="0.25μm SiGe",
        LO_RMS_jitter_fs=70,
        LO_linewidth_Hz=100e3,   # 100 kHz
        
        # ADC specs (28nm CMOS)
        ADC_technology="28nm CMOS", 
        ADC_ENOB=5.0,
        ADC_sampling_rate_Gsps=15
    ),
    
    # Same frame duration for coherence
    frame_duration_s=0.1e-6,    # 0.1 μs = 100 ns  
    signal_bandwidth_Hz=10e9    # 10 GHz
)

# Dictionary for easy profile access
HARDWARE_PROFILES = {
    "High_Performance": HIGH_PERFORMANCE_PROFILE,
    "SWaP_Efficient": SWAP_EFFICIENT_PROFILE
}

# =============================================================================
# SIMULATION CONTROL PARAMETERS
# =============================================================================
@dataclass
class SimulationControl:
    """Parameters controlling simulation execution."""
    
    # SNR range for simulations
    SNR_dB_min: float = -10     # Minimum SNR [dB]
    SNR_dB_max: float = 40      # Maximum SNR [dB]
    SNR_dB_points: int = 51     # Number of SNR points
    
    # Monte Carlo parameters
    n_monte_carlo: int = 1000   # Number of MC iterations
    n_pilots: int = 64          # Number of pilot symbols (M in manuscript)
    
    # Frequency sweep parameters  
    f_c_sweep_points: int = 5   # Number of carrier frequencies to test
    
    # Channel state parameters
    n_channel_realizations: int = 100  # Number of channel state realizations
    
    # Convergence parameters
    convergence_threshold: float = 1e-6
    max_iterations: int = 100
    
    @property
    def SNR_dB_array(self) -> np.ndarray:
        """Generate SNR array for simulations."""
        return np.linspace(self.SNR_dB_min, self.SNR_dB_max, self.SNR_dB_points)
    
    @property
    def SNR_linear_array(self) -> np.ndarray:
        """Generate linear SNR array."""
        return 10 ** (self.SNR_dB_array / 10)

# =============================================================================
# DERIVED PARAMETERS AND UTILITY FUNCTIONS
# =============================================================================
class DerivedParameters:
    """Calculate derived parameters from base configuration."""
    
    @staticmethod
    def path_loss_dB(distance_m: float, frequency_Hz: float) -> float:
        """Calculate free-space path loss [dB]."""
        lambda_c = PhysicalConstants.wavelength(frequency_Hz)
        return 20 * np.log10(4 * np.pi * distance_m / lambda_c)
    
    @staticmethod
    def link_budget_dB(tx_power_dBm: float, tx_gain_dBi: float, rx_gain_dBi: float,
                      distance_m: float, frequency_Hz: float) -> dict:
        """Calculate complete link budget."""
        path_loss = DerivedParameters.path_loss_dB(distance_m, frequency_Hz)
        rx_power_dBm = tx_power_dBm + tx_gain_dBi + rx_gain_dBi - path_loss
        
        return {
            'tx_power_dBm': tx_power_dBm,
            'tx_gain_dBi': tx_gain_dBi,
            'rx_gain_dBi': rx_gain_dBi,
            'path_loss_dB': path_loss,
            'rx_power_dBm': rx_power_dBm
        }
    
    @staticmethod
    def doppler_shift(velocity_ms: float, frequency_Hz: float) -> float:
        """Calculate Doppler shift [Hz]."""
        return frequency_Hz * velocity_ms / PhysicalConstants.c
    
    @staticmethod
    def thermal_noise_power(bandwidth_Hz: float, 
                           temperature_K: float = PhysicalConstants.T_noise,
                           noise_figure_dB: float = 10) -> float:
        """Calculate thermal noise power [W]."""
        noise_temp = temperature_K * (10**(noise_figure_dB/10))
        return PhysicalConstants.k * noise_temp * bandwidth_Hz
    
    @staticmethod
    def thermal_noise_power_dBm(bandwidth_Hz: float,
                               temperature_K: float = PhysicalConstants.T_noise,
                               noise_figure_dB: float = 10) -> float:
        """Calculate thermal noise power [dBm]."""
        noise_watts = DerivedParameters.thermal_noise_power(bandwidth_Hz, temperature_K, noise_figure_dB)
        return 10 * np.log10(noise_watts * 1000)  # Convert to dBm
    
    @staticmethod
    def capacity_ceiling(Gamma_eff: float, sigma_phi_sq: float) -> float:
        """
        Calculate hardware-limited capacity ceiling [bits/symbol].
        C_sat = log₂(1 + e^(-σ_φ²)/Γ_eff)
        """
        phase_factor = np.exp(-sigma_phi_sq)
        return np.log2(1 + phase_factor / Gamma_eff)

# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================
# Create default instances for easy import
scenario = ScenarioParameters()
simulation = SimulationControl()
constants = PhysicalConstants()
derived = DerivedParameters()

# =============================================================================
# CONFIGURATION VALIDATION AND LINK BUDGET CHECK
# =============================================================================
def validate_configuration():
    """Validate configuration parameters for consistency."""
    
    print("=== Configuration Validation ===\n")
    
    # Check link budget for different scenarios
    print("Link Budget Analysis:")
    print("-" * 70)
    
    scenarios = [
        ("Default (0.5m, 20dBm)", scenario.D_antenna, scenario.P_tx_dBm_default),
        ("Large Antenna (1m, 20dBm)", scenario.D_antenna_large, scenario.P_tx_dBm_default),
        ("High Power (0.5m, 30dBm)", scenario.D_antenna, scenario.P_tx_dBm_high),
        ("Both (1m, 30dBm)", scenario.D_antenna_large, scenario.P_tx_dBm_high)
    ]
    
    for name, diameter, tx_power in scenarios:
        print(f"\n{name}:")
        
        # Calculate gains
        tx_gain = scenario.antenna_gain_dB(diameter)
        rx_gain = tx_gain  # Same antenna
        
        # Calculate link budget
        budget = derived.link_budget_dB(
            tx_power, tx_gain, rx_gain,
            scenario.R_default, scenario.f_c_default
        )
        
        # Calculate noise floor
        noise_dBm = derived.thermal_noise_power_dBm(10e9, noise_figure_dB=10)
        
        # Link margin
        margin = budget['rx_power_dBm'] - noise_dBm
        
        print(f"  Tx Power: {tx_power:.1f} dBm")
        print(f"  Antenna Gain: {tx_gain:.1f} dBi (each)")
        print(f"  Path Loss: {budget['path_loss_dB']:.1f} dB")
        print(f"  Rx Power: {budget['rx_power_dBm']:.1f} dBm")
        print(f"  Noise Floor: {noise_dBm:.1f} dBm")
        print(f"  Link Margin: {margin:.1f} dB {'✓' if margin > 0 else '✗ INSUFFICIENT'}")
    
    print("\n" + "-" * 70)
    
    # Check hardware profiles
    print("\nHardware Profiles:")
    for name, profile in HARDWARE_PROFILES.items():
        print(f"\n{name}:")
        
        # Verify component contributions sum correctly
        component_sum = profile.Gamma_PA + profile.Gamma_LO + profile.Gamma_ADC
        relative_error = abs(component_sum - profile.Gamma_eff) / profile.Gamma_eff
        
        print(f"  Gamma_eff: {profile.Gamma_eff:.4f}")
        print(f"  Component sum: {component_sum:.4f} (error: {relative_error*100:.1f}%)")
        
        # Calculate and display phase noise variance
        sigma_phi_sq = profile.phase_noise_variance
        print(f"  Phase noise variance: {sigma_phi_sq:.4f} rad² (σ_φ = {np.sqrt(sigma_phi_sq):.3f} rad)")
        
        # Check coherence
        print(f"  Frame duration: {profile.frame_duration_s*1e6:.1f} μs")
        print(f"  Coherence time (for σ_φ²=0.1): {profile.coherence_time*1e6:.1f} μs")
        
        if sigma_phi_sq > 0.1:
            print(f"  ⚠️  WARNING: Frame duration may be too long for coherent processing!")
        else:
            print(f"  ✓ Frame duration suitable for coherent processing")
        
        # Calculate capacity ceiling
        ceiling = derived.capacity_ceiling(profile.Gamma_eff, sigma_phi_sq)
        print(f"  Capacity ceiling: {ceiling:.2f} bits/symbol")
    
    # Print recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("1. For positive link margin at 2000 km:")
    print("   - Use 1m antennas OR")
    print("   - Increase Tx power to 30 dBm OR")
    print("   - Use both for robust operation")
    print("2. SWaP_Efficient Gamma_eff corrected to 0.025 (was 0.045)")
    print("3. Frame duration of 100 ns is correct for 100 kHz linewidth")
    print("="*70 + "\n")

# Run validation on import
if __name__ == "__main__":
    validate_configuration()
    
    # Print summary
    print("\n=== THz ISL ISAC Simulation Configuration Summary ===")
    print(f"\nCarrier Frequency Range: {scenario.f_c_min/1e9:.0f} - {scenario.f_c_max/1e9:.0f} GHz")
    print(f"ISL Distance Range: {scenario.R_min/1e3:.0f} - {scenario.R_max/1e3:.0f} km")
    print(f"Default Operating Point: {scenario.f_c_default/1e9:.0f} GHz, {scenario.R_default/1e3:.0f} km")
    
    print("\n--- Hardware Profiles (CORRECTED) ---")
    for name, profile in HARDWARE_PROFILES.items():
        print(f"\n{name}:")
        print(f"  Gamma_eff: {profile.Gamma_eff:.4f} (EVM: {profile.EVM_total_percent:.1f}%)")
        print(f"  Phase noise variance: {profile.phase_noise_variance:.4f} rad²")
        print(f"  Frame duration: {profile.frame_duration_s*1e9:.0f} ns")
        print(f"  Capacity Ceiling: {derived.capacity_ceiling(profile.Gamma_eff, profile.phase_noise_variance):.2f} bits/symbol")