#!/usr/bin/env python3
"""
simulation_config.py

Central configuration file for THz LEO-ISL ISAC system simulations.
This module defines all physical constants, simulation parameters, and hardware profiles
to ensure consistency across all simulation scripts.

Based on:
1. "Fundamental Limits of THz Inter-Satellite ISAC Under Hardware Impairments" (Main manuscript)
2. "Deriving a Justifiable Range for the Hardware Quality Factor (Γ_eff) in THz LEO-ISL" (Hardware analysis)

Author: THz ISL ISAC Simulation Team
Date: 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
class PhysicalConstants:
    """Fundamental physical constants used in simulations."""
    
    c = 3e8                      # Speed of light [m/s]
    k = 1.380649e-23            # Boltzmann's constant [J/K]
    T_noise = 290               # Standard noise temperature [K]
    
    @classmethod
    def wavelength(cls, frequency_hz):
        """Calculate wavelength from frequency."""
        return cls.c / frequency_hz

# =============================================================================
# SCENARIO PARAMETERS
# =============================================================================
@dataclass
class ScenarioParameters:
    """Parameters defining the LEO ISL scenario."""
    
    # Carrier frequency range
    f_c_min: float = 100e9      # Minimum carrier frequency [Hz] (100 GHz)
    f_c_max: float = 600e9      # Maximum carrier frequency [Hz] (600 GHz)
    f_c_default: float = 300e9  # Default carrier frequency [Hz] (300 GHz)
    
    # ISL geometry
    R_min: float = 500e3        # Minimum ISL distance [m] (500 km)
    R_max: float = 5000e3       # Maximum ISL distance [m] (5000 km)
    R_default: float = 2000e3   # Default ISL distance [m] (2000 km)
    
    # Relative dynamics
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
# HARDWARE PROFILES
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
        """Calculate phase noise variance σ_φ² [rad²]."""
        # Based on Wiener process model: σ_φ² ≈ 2π * Δν * T
        return 2 * np.pi * self.components.LO_linewidth_Hz * self.frame_duration_s
    
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
    
    # System parameters
    frame_duration_s=100e-6,    # 100 μs
    signal_bandwidth_Hz=10e9    # 10 GHz
)

SWAP_EFFICIENT_PROFILE = HardwareProfile(
    name="SWaP_Efficient", 
    description="Silicon-based system optimized for SWaP and cost with DPD",
    
    # Aggregate quality factor  
    Gamma_eff=0.045,  # From detailed analysis (note: not 0.045)
    
    # Component contributions
    Gamma_PA=0.0438,          # CMOS with DPD, 20.93% EVM
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
    
    # System parameters
    frame_duration_s=100e-6,    # 100 μs
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
    def capacity_ceiling(Gamma_eff: float) -> float:
        """Calculate hardware-limited capacity ceiling [bits/symbol]."""
        return np.log2(1 + 1 / Gamma_eff)

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
    
    # Check hardware profiles
    for name, profile in HARDWARE_PROFILES.items():
        # Verify component contributions approximately sum to total
        component_sum = profile.Gamma_PA + profile.Gamma_LO + profile.Gamma_ADC
        relative_error = abs(component_sum - profile.Gamma_eff) / profile.Gamma_eff
        
        if relative_error > 0.1:  # Allow 10% discrepancy
            print(f"Warning: {name} profile component sum ({component_sum:.6f}) "
                  f"differs from Gamma_eff ({profile.Gamma_eff:.6f}) by {relative_error*100:.1f}%")
    
    # Check scenario parameters
    assert scenario.R_min < scenario.R_default < scenario.R_max, "Invalid distance range"
    assert scenario.f_c_min < scenario.f_c_default < scenario.f_c_max, "Invalid frequency range"
    
    print("Configuration validation complete.")

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
        print(f"  Capacity Ceiling: {derived.capacity_ceiling(profile.Gamma_eff):.2f} bits/symbol")
        print(f"  Phase Noise Variance: {profile.phase_noise_variance:.4f} rad²")