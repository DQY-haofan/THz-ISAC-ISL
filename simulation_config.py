#!/usr/bin/env python3
"""
simulation_config.py - Enhanced version with extended hardware profiles
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

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
    f_c_max: float = 1000e9     # Maximum carrier frequency [Hz] (1 THz) - Extended!
    f_c_default: float = 300e9  # Default carrier frequency [Hz] (300 GHz)
    
    # ISL geometry [meters]
    R_min: float = 500e3        # Minimum ISL distance [m] (500 km)
    R_max: float = 5000e3       # Maximum ISL distance [m] (5000 km)
    R_default: float = 2000e3   # Default ISL distance [m] (2000 km)
    
    # Relative dynamics [m/s]
    v_rel_max: float = 15e3     # Maximum relative velocity [m/s] (15 km/s)
    v_rel_default: float = 10e3 # Default relative velocity [m/s] (10 km/s)
    a_rel_max: float = 100      # Maximum relative acceleration [m/s²]
    
    # Enhanced antenna parameters for THz
    antenna_options: Dict[str, float] = None  # Will be set in __post_init__
    
    # Enhanced transmit power options
    power_options: Dict[str, float] = None    # Will be set in __post_init__
    
    def __post_init__(self):
        """Initialize dictionaries after dataclass creation."""
        self.antenna_options = {
            "small": 0.3,      # 30 cm - compact
            "medium": 0.5,     # 50 cm - baseline
            "large": 1.0,      # 1 m - enhanced
            "xlarge": 2.0      # 2 m - maximum performance
        }
        
        self.power_options = {
            "low": 10,         # 10 dBm (10 mW)
            "medium": 20,      # 20 dBm (100 mW)
            "high": 30,        # 30 dBm (1 W)
            "maximum": 33      # 33 dBm (2 W) - practical limit
        }
    
    # Antenna efficiency
    eta_antenna: float = 0.65   # Enhanced for THz
    
    # Derived antenna parameters
    def antenna_gain(self, diameter: float = None, frequency_hz: float = None) -> float:
        """Calculate antenna gain [linear] for given diameter and frequency."""
        if diameter is None:
            diameter = self.antenna_options["medium"]
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
            diameter = self.antenna_options["medium"]
        lambda_c = PhysicalConstants.wavelength(frequency_hz)
        return 1.02 * lambda_c / diameter
    
    def beam_rolloff_factor(self, frequency_hz: float, diameter: float = None) -> float:
        """Calculate beam rolloff factor γ = 2.77/θ_3dB² [rad⁻²]."""
        theta_3dB = self.beamwidth_3dB(frequency_hz, diameter)
        return 2.77 / (theta_3dB ** 2)

# =============================================================================
# EXTENDED HARDWARE PROFILES
# =============================================================================
@dataclass
class HardwareComponentSpecs:
    """Specifications for individual hardware components."""
    
    # Power Amplifier
    PA_technology: str          # Technology type
    PA_EVM_percent: float       # PA Error Vector Magnitude [%]
    PA_P_sat_dBm: float        # PA saturation power [dBm]
    PA_efficiency: float        # Power-added efficiency
    
    # Local Oscillator / PLL
    LO_technology: str          # Technology type
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
        delta_nu = self.components.LO_linewidth_Hz
        T = self.frame_duration_s
        variance = (4/3) * np.pi * delta_nu * T
        return variance
    
    @property
    def coherence_time(self) -> float:
        """Estimate coherence time for σ_φ² ≈ 0.1 rad²."""
        delta_nu = self.components.LO_linewidth_Hz
        return 0.1 / ((4/3) * np.pi * delta_nu)
    
    @property
    def EVM_total_percent(self) -> float:
        """Total system EVM [%]."""
        return 100 * np.sqrt(self.Gamma_eff)

# Extended hardware profiles with more options
HARDWARE_PROFILES = {
    # State-of-the-art profiles
    "State_of_Art": HardwareProfile(
        name="State_of_Art",
        description="Best available technology (future projection)",
        Gamma_eff=0.005,  # -23 dB EVM
        Gamma_PA=0.0045,
        Gamma_LO=2e-7,
        Gamma_ADC=8e-5,
        components=HardwareComponentSpecs(
            PA_technology="Advanced InP HBT",
            PA_EVM_percent=7.1,
            PA_P_sat_dBm=18,
            PA_efficiency=0.20,
            LO_technology="Photonic Integration",
            LO_RMS_jitter_fs=10,
            LO_linewidth_Hz=10e3,  # 10 kHz
            ADC_technology="7nm CMOS",
            ADC_ENOB=7.0,
            ADC_sampling_rate_Gsps=40
        ),
        frame_duration_s=1e-6,  # 1 μs
        signal_bandwidth_Hz=50e9  # 50 GHz
    ),
    
    # Original profiles
    "High_Performance": HardwareProfile(
        name="High_Performance",
        description="III-V semiconductor based system",
        Gamma_eff=0.01,
        Gamma_PA=0.009,
        Gamma_LO=4.3e-7,
        Gamma_ADC=1.7e-4,
        components=HardwareComponentSpecs(
            PA_technology="InP DHBT",
            PA_EVM_percent=10.6,
            PA_P_sat_dBm=15,
            PA_efficiency=0.15,
            LO_technology="28nm CMOS",
            LO_RMS_jitter_fs=20.9,
            LO_linewidth_Hz=100e3,
            ADC_technology="20nm CMOS",
            ADC_ENOB=5.95,
            ADC_sampling_rate_Gsps=20
        ),
        frame_duration_s=0.1e-6,
        signal_bandwidth_Hz=10e9
    ),
    
    "SWaP_Efficient": HardwareProfile(
        name="SWaP_Efficient",
        description="Silicon-based system with DPD",
        Gamma_eff=0.025,
        Gamma_PA=0.022,
        Gamma_LO=4.8e-6,
        Gamma_ADC=6.5e-4,
        components=HardwareComponentSpecs(
            PA_technology="SiGe BiCMOS with DPD",
            PA_EVM_percent=14.9,
            PA_P_sat_dBm=10,
            PA_efficiency=0.05,
            LO_technology="0.25μm SiGe",
            LO_RMS_jitter_fs=70,
            LO_linewidth_Hz=100e3,
            ADC_technology="28nm CMOS",
            ADC_ENOB=5.0,
            ADC_sampling_rate_Gsps=15
        ),
        frame_duration_s=0.1e-6,
        signal_bandwidth_Hz=10e9
    ),
    
    # Low-cost profile
    "Low_Cost": HardwareProfile(
        name="Low_Cost",
        description="Budget silicon solution",
        Gamma_eff=0.05,  # -13 dB EVM
        Gamma_PA=0.045,
        Gamma_LO=1e-5,
        Gamma_ADC=0.001,
        components=HardwareComponentSpecs(
            PA_technology="65nm CMOS",
            PA_EVM_percent=22.4,
            PA_P_sat_dBm=7,
            PA_efficiency=0.02,
            LO_technology="65nm CMOS",
            LO_RMS_jitter_fs=150,
            LO_linewidth_Hz=500e3,
            ADC_technology="65nm CMOS",
            ADC_ENOB=4.0,
            ADC_sampling_rate_Gsps=10
        ),
        frame_duration_s=0.05e-6,  # Shorter to manage phase noise
        signal_bandwidth_Hz=5e9
    ),
    
    # Custom profile for parameter sweeps
    "Custom": HardwareProfile(
        name="Custom",
        description="Configurable profile for sweeps",
        Gamma_eff=0.01,  # Will be modified during sweeps
        Gamma_PA=0.009,
        Gamma_LO=4.3e-7,
        Gamma_ADC=1.7e-4,
        components=HardwareComponentSpecs(
            PA_technology="Variable",
            PA_EVM_percent=10.0,
            PA_P_sat_dBm=15,
            PA_efficiency=0.10,
            LO_technology="Variable",
            LO_RMS_jitter_fs=50,
            LO_linewidth_Hz=100e3,
            ADC_technology="Variable",
            ADC_ENOB=6.0,
            ADC_sampling_rate_Gsps=20
        ),
        frame_duration_s=0.1e-6,
        signal_bandwidth_Hz=10e9
    )
}

# =============================================================================
# SIMULATION CONTROL PARAMETERS
# =============================================================================
@dataclass
class SimulationControl:
    """Parameters controlling simulation execution."""
    
    # SNR range for simulations
    SNR_dB_min: float = -10     # Minimum SNR [dB]
    SNR_dB_max: float = 50      # Maximum SNR [dB] - Extended!
    SNR_dB_points: int = 61     # Number of SNR points
    
    # Monte Carlo parameters
    n_monte_carlo: int = 1000   # Number of MC iterations
    n_pilots: int = 64          # Number of pilot symbols (M in manuscript)
    
    # Frequency sweep parameters  
    f_c_sweep_points: int = 7   # More frequency points
    
    # Hardware parameter sweep
    gamma_eff_sweep: List[float] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Initialize sweep arrays."""
        # Logarithmic sweep of hardware quality factor
        self.gamma_eff_sweep = np.logspace(-3, -1, 20).tolist()  # 0.001 to 0.1
    
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
        return 10 * np.log10(noise_watts * 1000)
    
    @staticmethod
    def capacity_ceiling(Gamma_eff: float, sigma_phi_sq: float) -> float:
        """Calculate hardware-limited capacity ceiling [bits/symbol]."""
        phase_factor = np.exp(-sigma_phi_sq)
        return np.log2(1 + phase_factor / Gamma_eff)
    
    @staticmethod
    def find_snr_for_hardware_limit(Gamma_eff: float, target_ratio: float = 0.95) -> float:
        """Find SNR where capacity reaches target_ratio of hardware ceiling."""
        # For hardware-limited regime: SNR ≈ 1/Gamma_eff
        # We want SNR where capacity = target_ratio * ceiling
        # This occurs roughly when SNR = (1/Gamma_eff) / (1 - target_ratio)
        return 10 * np.log10((1/Gamma_eff) / (1 - target_ratio))

# =============================================================================
# OBSERVABLE PARAMETERS FOR SINGLE ISL
# =============================================================================
class ObservableParameters:
    """Define what can be observed with single ISL."""
    
    # Single ISL can only observe:
    # 1. Range (radial distance)
    # 2. Range-rate (radial velocity)
    # Cannot observe: cross-track position/velocity, pointing errors
    
    observable_params = ["range", "range_rate"]
    unobservable_params = ["cross_track_position", "cross_track_velocity", "pointing_errors"]
    
    @staticmethod
    def get_observable_dimension():
        """Return dimension of observable parameter space."""
        return len(ObservableParameters.observable_params)
    
    @staticmethod
    def print_observability_warning():
        """Print warning about single ISL limitations."""
        print("\n" + "="*70)
        print("WARNING: Single ISL Observability Limitations")
        print("="*70)
        print("Observable parameters (2):")
        for param in ObservableParameters.observable_params:
            print(f"  ✓ {param}")
        print("\nUnobservable parameters (6):")
        for param in ObservableParameters.unobservable_params:
            print(f"  ✗ {param}")
        print("\nNote: Full 3D state estimation requires multiple non-coplanar ISLs")
        print("="*70 + "\n")

# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================
scenario = ScenarioParameters()
simulation = SimulationControl()
constants = PhysicalConstants()
derived = DerivedParameters()
observable = ObservableParameters()

# Print observability warning on import
if __name__ == "__main__":
    observable.print_observability_warning()