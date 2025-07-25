#!/usr/bin/env python3
"""
simulation_config.py - CORRECTED VERSION WITH COHERENT PARAMETERS

Fixed the frame duration to ensure physical coherence.
Key insight: For a 100 kHz linewidth oscillator, we need much shorter
processing frames to maintain coherence.

Author: THz ISL ISAC Simulation Team
Date: 2024
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
    D_antenna: float = 0.5      # Antenna diameter [m]
    eta_antenna: float = 0.55   # Antenna efficiency
    
    # Derived antenna parameters
    @property
    def antenna_gain(self) -> float:
        """Calculate antenna gain at default frequency [linear]."""
        lambda_c = PhysicalConstants.wavelength(self.f_c_default)
        return self.eta_antenna * (np.pi * self.D_antenna / lambda_c) ** 2
    
    @property
    def antenna_gain_dB(self) -> float:
        """Antenna gain in dB."""
        return 10 * np.log10(self.antenna_gain)
    
    def beamwidth_3dB(self, frequency_hz: float) -> float:
        """Calculate 3dB beamwidth [rad] at given frequency."""
        lambda_c = PhysicalConstants.wavelength(frequency_hz)
        return 1.02 * lambda_c / self.D_antenna
    
    def beam_rolloff_factor(self, frequency_hz: float) -> float:
        """Calculate beam rolloff factor γ = 2.77/θ_3dB² [rad⁻²]."""
        theta_3dB = self.beamwidth_3dB(frequency_hz)
        return 2.77 / (theta_3dB ** 2)

# =============================================================================
# HARDWARE PROFILES - WITH COHERENT FRAME DURATION
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
    
    # System parameters - CORRECTED FOR COHERENCE
    frame_duration_s: float    # Frame duration [seconds]
    signal_bandwidth_Hz: float # Signal bandwidth [Hz]
    
    @property
    def phase_noise_variance(self) -> float:
        """
        Calculate phase noise variance σ_φ² [rad²].
        Using Wiener process approximation for short frames.
        σ_φ² ≈ (4/3) * π * Δν * T
        
        CRITICAL: For coherent processing, we need σ_φ² << 1
        """
        delta_nu = self.components.LO_linewidth_Hz
        T = self.frame_duration_s
        
        # Calculate phase noise variance
        variance = (4/3) * np.pi * delta_nu * T
        
        # Print warning if variance is too large
        if variance > 1.0:
            print(f"WARNING: Phase noise variance {variance:.2f} rad² is too large!")
            print(f"  Consider reducing frame duration or improving oscillator")
        
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
    
    # Aggregate quality factor
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
    
    # CORRECTED: Much shorter frame for coherence
    # For 100 kHz linewidth and target σ_φ² ≈ 0.042:
    # T = 0.042 / ((4/3) * π * 1e5) ≈ 0.1 μs
    frame_duration_s=0.1e-6,    # 0.1 μs = 100 ns
    signal_bandwidth_Hz=10e9    # 10 GHz
)

SWAP_EFFICIENT_PROFILE = HardwareProfile(
    name="SWaP_Efficient", 
    description="Silicon-based system optimized for SWaP and cost with DPD",
    
    # Aggregate quality factor  
    Gamma_eff=0.025,  # From detailed analysis
    
    # Component contributions
    Gamma_PA=0.0238,          # CMOS with DPD - adjusted for consistency
    Gamma_LO=4.8e-6,          # Based on 70 fs RMS jitter
    Gamma_ADC=6.5e-4,         # Based on 5.0 ENOB
    
    # Component specifications
    components=HardwareComponentSpecs(
        # PA specs (CMOS with DPD)
        PA_technology="CMOS with DPD",
        PA_EVM_percent=20.93,   # Post-DPD performance
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
    
    # CORRECTED: Same short frame duration for coherence
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
    def doppler_shift(velocity_ms: float, frequency_Hz: float) -> float:
        """Calculate Doppler shift [Hz]."""
        return frequency_Hz * velocity_ms / PhysicalConstants.c
    
    @staticmethod
    def thermal_noise_power(bandwidth_Hz: float, 
                           temperature_K: float = PhysicalConstants.T_noise) -> float:
        """Calculate thermal noise power [W]."""
        return PhysicalConstants.k * temperature_K * bandwidth_Hz
    
    @staticmethod
    def capacity_ceiling(Gamma_eff: float, sigma_phi_sq: float) -> float:
        """
        Calculate hardware-limited capacity ceiling [bits/symbol].
        For complex baseband channel (no 1/2 factor).
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
# CONFIGURATION VALIDATION
# =============================================================================
def validate_configuration():
    """Validate configuration parameters for consistency."""
    
    print("=== Configuration Validation ===\n")
    
    # Check hardware profiles
    for name, profile in HARDWARE_PROFILES.items():
        print(f"{name}:")
        
        # Verify component contributions approximately sum to total
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
            print(f"  ⚠️  WARNING: Frame duration too long for coherent processing!")
        else:
            print(f"  ✓ Frame duration suitable for coherent processing")
        
        # Calculate capacity ceiling
        ceiling = derived.capacity_ceiling(profile.Gamma_eff, sigma_phi_sq)
        print(f"  Capacity ceiling: {ceiling:.2f} bits/symbol")
        print()
    
    # Check scenario parameters
    print("Scenario Parameters:")
    print(f"  Distance range: {scenario.R_min/1e3:.0f} - {scenario.R_max/1e3:.0f} km")
    print(f"  Velocity range: 0 - {scenario.v_rel_max/1e3:.0f} km/s")
    print(f"  All parameters in SI units ✓")
    
    # Print critical insight
    print("\n" + "="*60)
    print("CRITICAL INSIGHT:")
    print("For THz ISL with practical oscillators (100 kHz linewidth),")
    print("coherent processing requires extremely short frames (~100 ns).")
    print("This is a fundamental constraint of THz hardware!")
    print("="*60 + "\n")

# Run validation on import
if __name__ == "__main__":
    validate_configuration()
    
    # Print summary
    print("\n=== THz ISL ISAC Simulation Configuration Summary ===")
    print(f"\nCarrier Frequency Range: {scenario.f_c_min/1e9:.0f} - {scenario.f_c_max/1e9:.0f} GHz")
    print(f"ISL Distance Range: {scenario.R_min/1e3:.0f} - {scenario.R_max/1e3:.0f} km")
    print(f"Default Operating Point: {scenario.f_c_default/1e9:.0f} GHz, {scenario.R_default/1e3:.0f} km")
    
    print("\n--- Hardware Profiles ---")
    for name, profile in HARDWARE_PROFILES.items():
        print(f"\n{name}:")
        print(f"  Gamma_eff: {profile.Gamma_eff:.4f} (EVM: {profile.EVM_total_percent:.1f}%)")
        print(f"  Phase noise variance: {profile.phase_noise_variance:.4f} rad²")
        print(f"  Frame duration: {profile.frame_duration_s*1e9:.0f} ns")
        print(f"  Capacity Ceiling: {derived.capacity_ceiling(profile.Gamma_eff, profile.phase_noise_variance):.2f} bits/symbol")