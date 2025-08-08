#!/usr/bin/env python3
"""
simulation_config.py - IEEE Publication Style Configuration
Updated with 1THz support and enhanced parameters
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import os
import json

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

# =============================================================================
# IEEE PUBLICATION STYLE CONFIGURATION
# =============================================================================
class IEEEStyle:
    """IEEE publication style settings for all plots."""
    
    # Font sizes (adjusted for better proportions)
    FONT_SIZES = {
        'title': 14,        # Figure title
        'label': 12,        # Axis labels
        'tick': 10,         # Tick labels
        'legend': 10,       # Legend text
        'annotation': 9,    # Annotations
        'text': 10,         # General text
    }
    
    # Fixed figure sizes for consistency
    FIG_SIZES = {
        'single': (6, 4.5),      # Standard single figure
        'double': (12, 4.5),     # Double column figure
        'square': (5, 5),        # Square figure
        'large': (7, 5),         # Large single figure
        '3d': (8, 6),            # 3D plots
    }
    
    # Line and marker properties
    LINE_PROPS = {
        'linewidth': 2.0,
        'markersize': 6,
        'markeredgewidth': 1.2,
        'markeredgecolor': 'white',
    }
    
    # Grid properties
    GRID_PROPS = {
        'alpha': 0.3,
        'linestyle': ':',
        'linewidth': 0.8,
    }
    
    # Professional color schemes - UPDATED
    COLORS_PROFESSIONAL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    COLORS_FEASIBILITY = {
        'infeasible': '#d0d0d0',      # Light gray (was #f0f0f0)
        'comm_only': '#ffd92f',        # Yellow (unchanged)
        'sense_only': '#87ceeb',       # Sky blue (was #4daf4a)
        'both': '#1f77b4',             # Blue (unchanged)
        'excellent': '#2ca02c'         # GREEN (was #d62728 red) - FIXED!
    }
    
    @staticmethod
    def setup():
        """Setup matplotlib for IEEE publication style."""
        # Use LaTeX-like fonts
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        
        # Set default font sizes
        plt.rcParams['font.size'] = IEEEStyle.FONT_SIZES['text']
        plt.rcParams['axes.titlesize'] = IEEEStyle.FONT_SIZES['title']
        plt.rcParams['axes.labelsize'] = IEEEStyle.FONT_SIZES['label']
        plt.rcParams['xtick.labelsize'] = IEEEStyle.FONT_SIZES['tick']
        plt.rcParams['ytick.labelsize'] = IEEEStyle.FONT_SIZES['tick']
        plt.rcParams['legend.fontsize'] = IEEEStyle.FONT_SIZES['legend']
        
        # Set line and marker defaults
        plt.rcParams['lines.linewidth'] = IEEEStyle.LINE_PROPS['linewidth']
        plt.rcParams['lines.markersize'] = IEEEStyle.LINE_PROPS['markersize']
        
        # Set figure properties
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.05
        
        # Set axes properties
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = IEEEStyle.GRID_PROPS['alpha']
        plt.rcParams['grid.linestyle'] = IEEEStyle.GRID_PROPS['linestyle']
        
        # Legend properties
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['legend.edgecolor'] = 'black'
        plt.rcParams['legend.fancybox'] = False
        
        # Tick properties
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['ytick.major.size'] = 4
        
        # Remove top and right spines
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.right'] = True
    
    @staticmethod
    def get_colors():
        """Get IEEE-friendly color palette."""
        return IEEEStyle.COLORS_PROFESSIONAL
    
    @staticmethod
    def get_markers():
        """Get distinct markers for different data series."""
        return ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    @staticmethod
    def get_linestyles():
        """Get line styles for different data series."""
        return ['-', '--', '-.', ':', '-', '--', '-.', ':']

# =============================================================================
# DATA SAVING UTILITIES
# =============================================================================
class DataSaver:
    """Utility class for saving numerical data."""
    
    
    @staticmethod
    def save_data(filename: str, data: Dict, description: str = ""):
        """Add data to global diagnostics instead of separate files."""
        from diagnostics import diagnostics
        
        # Add to diagnostics results
        diagnostics.results[filename] = {
            'description': description,
            'data': data
        }
        
        # Extract key metrics
        for key, value in data.items():
            if 'capacity' in key.lower() and isinstance(value, (int, float)):
                diagnostics.add_key_metric("Summary", f"{filename}_{key}", value)
            elif 'rmse' in key.lower() and isinstance(value, (int, float)):
                diagnostics.add_key_metric("Summary", f"{filename}_{key}", value)
        
        print(f"  Data added to comprehensive report: {filename}")

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
    
    # Extended carrier frequency range to 1 THz
    f_c_min: float = 100e9      # Minimum carrier frequency [Hz] (100 GHz)
    f_c_max: float = 1000e9     # Maximum carrier frequency [Hz] (1 THz)
    f_c_default: float = 300e9  # Default carrier frequency [Hz] (300 GHz)
    
    # ISL geometry [meters]
    R_min: float = 500e3        # Minimum ISL distance [m] (500 km)
    R_max: float = 5000e3       # Maximum ISL distance [m] (5000 km)
    R_default: float = 2000e3   # Default ISL distance [m] (2000 km)
    
    # Relative dynamics [m/s]
    v_rel_max: float = 15e3     # Maximum relative velocity [m/s] (15 km/s)
    v_rel_default: float = 10e3 # Default relative velocity [m/s] (10 km/s)
    a_rel_max: float = 100      # Maximum relative acceleration [m/s²]
    
    # Pointing error parameters
    pointing_error_rms_rad: float = 1e-6  # RMS pointing error [rad] (1 µrad)
    
    # Enhanced antenna parameters for THz
    antenna_options: Dict[str, float] = None
    power_options: Dict[str, float] = None
    
    # Enhanced default values for THz
    default_tx_power_dBm: float = 33    # 33 dBm (2W) - high power for THz
    default_antenna_diameter: float = 1.5  # 1.5m - large antenna for THz
    
    def __post_init__(self):
        """Initialize dictionaries after dataclass creation."""
        self.antenna_options = {
            "small": 0.5,      # 50 cm - minimum practical
            "medium": 1.0,     # 1 m - baseline
            "large": 1.5,      # 1.5 m - enhanced
            "xlarge": 2.0      # 2 m - maximum performance
        }
        
        self.power_options = {
            "low": 20,         # 20 dBm (100 mW)
            "medium": 27,      # 27 dBm (500 mW)
            "high": 30,        # 30 dBm (1 W)
            "maximum": 33      # 33 dBm (2 W)
        }
    
    # Antenna efficiency
    eta_antenna: float = 0.65
    
    def antenna_gain(self, diameter: float = None, frequency_hz: float = None) -> float:
        """Calculate antenna gain [linear] for given diameter and frequency."""
        if diameter is None:
            diameter = self.default_antenna_diameter
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
            diameter = self.default_antenna_diameter
        lambda_c = PhysicalConstants.wavelength(frequency_hz)
        return 1.02 * lambda_c / diameter
    
    def beam_rolloff_factor(self, frequency_hz: float, diameter: float = None) -> float:
        """Calculate beam rolloff factor γ = 2.77/θ_3dB² [rad⁻²]."""
        theta_3dB = self.beamwidth_3dB(frequency_hz, diameter)
        return 2.77 / (theta_3dB ** 2)
    
    def calculate_pointing_loss_factor(self, frequency_hz: float, diameter: float = None, 
                                 pointing_error_rms: float = None) -> float:
        """Calculate expected pointing loss factor due to random pointing errors.
        
        Args:
            frequency_hz: Carrier frequency [Hz]
            diameter: Antenna diameter [m]
            pointing_error_rms: RMS pointing error [rad] - if None, use default
            
        Returns:
            Expected pointing loss factor (power)
        """
        if pointing_error_rms is None:
            pointing_error_rms = self.pointing_error_rms_rad
        
        if diameter is None:
            diameter = self.default_antenna_diameter
            
        gamma = self.beam_rolloff_factor(frequency_hz, diameter)
        # Expected value of exp(-gamma * ||theta_e||^2) where ||theta_e||^2 ~ chi^2(2)
        return 1 / (1 + 2 * gamma * pointing_error_rms**2)
    
    def sample_pointing_loss(self, frequency_hz: float, diameter: float = None,
                        pointing_error_rms: float = None, n_samples: int = 1) -> np.ndarray:
        """Sample random pointing loss factors.
        
        Args:
            frequency_hz: Carrier frequency [Hz]
            diameter: Antenna diameter [m]
            pointing_error_rms: RMS pointing error [rad] - if None, use default
            n_samples: Number of samples to generate
            
        Returns:
            Array of pointing loss factors (power)
        """
        if pointing_error_rms is None:
            pointing_error_rms = self.pointing_error_rms_rad
        
        if diameter is None:
            diameter = self.default_antenna_diameter
            
        gamma = self.beam_rolloff_factor(frequency_hz, diameter)
        
        # Sample 2D Gaussian pointing errors with specified RMS
        theta_x = np.random.normal(0, pointing_error_rms, n_samples)
        theta_y = np.random.normal(0, pointing_error_rms, n_samples)
        theta_squared = theta_x**2 + theta_y**2
        
        # Calculate pointing loss
        return np.exp(-gamma * theta_squared)

# =============================================================================
# HARDWARE PROFILES
# =============================================================================
@dataclass
class HardwareComponentSpecs:
    """Specifications for individual hardware components."""
    
    PA_technology: str
    PA_EVM_percent: float
    PA_P_sat_dBm: float
    PA_efficiency: float
    
    LO_technology: str
    LO_RMS_jitter_fs: float
    LO_linewidth_Hz: float
    
    ADC_technology: str
    ADC_ENOB: float
    ADC_sampling_rate_Gsps: float

@dataclass
class HardwareProfile:
    """Complete hardware profile with quality factors."""
    
    name: str
    description: str
    Gamma_eff: float
    Gamma_PA: float
    Gamma_LO: float
    Gamma_ADC: float
    components: HardwareComponentSpecs
    frame_duration_s: float
    signal_bandwidth_Hz: float
    
    @property
    def phase_noise_variance(self) -> float:
        """Calculate phase noise variance σ_φ² [rad²] for Wiener phase process.
        
        For oscillator with 3-dB linewidth Δν (Hz), the phase variance 
        accumulated over time T is: σ_φ² = 2π·Δν·T
        
        This is the standard Wiener process model where variance grows linearly with time.
        Reference: Standard phase noise theory in coherent optical/RF systems
        """
        delta_nu = self.components.LO_linewidth_Hz  # 3-dB linewidth in Hz
        T = self.frame_duration_s  # Observation/frame duration
        
        # Standard Wiener phase noise variance formula
        variance = 2 * np.pi * delta_nu * T
        
        # Add reasonable bounds check
        if variance > 1.0:
            print(f"Warning: Very large phase noise variance σ_φ²={variance:.3f} rad²")
            print(f"  (Δν={delta_nu/1e3:.1f} kHz, T={T*1e6:.1f} μs)")
            print(f"  Consider shorter frame duration or better oscillator")
        
        return variance
    
    @property
    def coherence_time(self) -> float:
        """Estimate coherence time for σ_φ² ≈ 0.1 rad²."""
        delta_nu = self.components.LO_linewidth_Hz
        return 0.1 / (2 * np.pi * delta_nu)
    
    @property
    def EVM_total_percent(self) -> float:
        """Total system EVM [%]."""
        return 100 * np.sqrt(self.Gamma_eff)

# Hardware profiles with updated parameters for THz
HARDWARE_PROFILES = {
    "State_of_Art": HardwareProfile(
        name="State_of_Art",
        description="Best available technology for THz",
        Gamma_eff=0.005,
        Gamma_PA=0.0045,
        Gamma_LO=2e-7,
        Gamma_ADC=8e-5,
        components=HardwareComponentSpecs(
            PA_technology="Advanced InP HBT",
            PA_EVM_percent=7.1,
            PA_P_sat_dBm=20,  # Higher for THz
            PA_efficiency=0.20,
            LO_technology="Photonic Integration",
            LO_RMS_jitter_fs=10,
            LO_linewidth_Hz=10e3,
            ADC_technology="7nm CMOS",
            ADC_ENOB=7.0,
            ADC_sampling_rate_Gsps=100  # Higher for 1THz
        ),
        frame_duration_s=1e-6,
        signal_bandwidth_Hz=100e9  # 100 GHz bandwidth
    ),
    
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
            PA_P_sat_dBm=18,
            PA_efficiency=0.15,
            LO_technology="28nm CMOS",
            LO_RMS_jitter_fs=20.9,
            LO_linewidth_Hz=100e3,
            ADC_technology="20nm CMOS",
            ADC_ENOB=5.95,
            ADC_sampling_rate_Gsps=50
        ),
        frame_duration_s=0.1e-6,
        signal_bandwidth_Hz=20e9
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
            PA_P_sat_dBm=15,
            PA_efficiency=0.05,
            LO_technology="0.25μm SiGe",
            LO_RMS_jitter_fs=70,
            LO_linewidth_Hz=100e3,
            ADC_technology="28nm CMOS",
            ADC_ENOB=5.0,
            ADC_sampling_rate_Gsps=20
        ),
        frame_duration_s=0.1e-6,
        signal_bandwidth_Hz=10e9
    ),
    
    "Low_Cost": HardwareProfile(
        name="Low_Cost",
        description="Budget silicon solution",
        Gamma_eff=0.05,
        Gamma_PA=0.045,
        Gamma_LO=1e-5,
        Gamma_ADC=0.001,
        components=HardwareComponentSpecs(
            PA_technology="65nm CMOS",
            PA_EVM_percent=22.4,
            PA_P_sat_dBm=10,
            PA_efficiency=0.02,
            LO_technology="65nm CMOS",
            LO_RMS_jitter_fs=150,
            LO_linewidth_Hz=500e3,
            ADC_technology="65nm CMOS",
            ADC_ENOB=4.0,
            ADC_sampling_rate_Gsps=10
        ),
        frame_duration_s=0.05e-6,
        signal_bandwidth_Hz=5e9
    ),
    
    "Custom": HardwareProfile(
        name="Custom",
        description="Configurable profile",
        Gamma_eff=0.01,
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
# SIMULATION CONTROL
# =============================================================================
@dataclass
class SimulationControl:
    """Parameters controlling simulation execution."""
    
    SNR_dB_min: float = -10
    SNR_dB_max: float = 60
    SNR_dB_points: int = 71
    
    default_SNR_dB: float = 40
    
    n_monte_carlo: int = 1000
    n_pilots: int = 64
    
    # DSE residual factor (parameterized)
    kappa_DSE: float = 1e-3  # DSE residual as fraction of thermal noise
    
    f_c_sweep_points: int = 10
    gamma_eff_sweep: List[float] = None
    pointing_error_sweep: List[float] = None
    
    def __post_init__(self):
        """Initialize sweep arrays."""
        self.gamma_eff_sweep = np.logspace(-3, -1, 20).tolist()
        self.pointing_error_sweep = [0.5e-6, 1e-6, 2e-6]
    
    @property
    def SNR_dB_array(self) -> np.ndarray:
        """Generate SNR array for simulations."""
        return np.linspace(self.SNR_dB_min, self.SNR_dB_max, self.SNR_dB_points)
    
    @property
    def SNR_linear_array(self) -> np.ndarray:
        """Generate linear SNR array."""
        return 10 ** (self.SNR_dB_array / 10)
    
    @property
    def frequency_sweep_GHz(self) -> np.ndarray:
        """Generate frequency sweep including 1THz."""
        return np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

# =============================================================================
# DERIVED PARAMETERS
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
    def find_snr_for_hardware_limit(Gamma_eff: float, sigma_phi_sq: float, target_ratio: float = 0.95) -> float:
        """Find SNR where capacity reaches target_ratio of hardware ceiling.
        
        Model: SINR_eff = SNR / (1 + Gamma_eff * exp(sigma_phi_sq) * SNR)
        Target: C = log2(1 + SINR_eff) = target_ratio * log2(1 + exp(-sigma_phi_sq)/Gamma_eff)
        Solution: SNR = (target_ratio / (1 - target_ratio)) * exp(sigma_phi_sq) / Gamma_eff
        
        Args:
            Gamma_eff: Hardware quality factor
            sigma_phi_sq: Phase noise variance [rad²]
            target_ratio: Target ratio of ceiling (default 0.95)
            
        Returns:
            SNR in dB where capacity reaches target_ratio of ceiling
        """
        if target_ratio >= 1.0 or target_ratio <= 0:
            raise ValueError("target_ratio must be in (0, 1)")
            
        snr_linear = (target_ratio / (1.0 - target_ratio)) * np.exp(sigma_phi_sq) / Gamma_eff
        return 10.0 * np.log10(snr_linear)

# =============================================================================
# OBSERVABLE PARAMETERS
# =============================================================================
class ObservableParameters:
    """Define what can be observed with single ISL."""
    
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
# DEFAULT INSTANCES
# =============================================================================
scenario = ScenarioParameters()
simulation = SimulationControl()
constants = PhysicalConstants()
derived = DerivedParameters()
observable = ObservableParameters()
data_saver = DataSaver()

# Setup IEEE style on import
IEEEStyle.setup()

if __name__ == "__main__":
    print("IEEE Style Configuration Loaded")
    print(f"Default font sizes: {IEEEStyle.FONT_SIZES}")
    print(f"Available figure sizes: {list(IEEEStyle.FIG_SIZES.keys())}")
    print(f"Phase noise model: σ_φ² = 2π × Δν × T")
    print(f"Pointing error RMS: {scenario.pointing_error_rms_rad*1e6:.1f} µrad")
    print(f"Default TX power: {scenario.default_tx_power_dBm} dBm")
    print(f"Default antenna: {scenario.default_antenna_diameter} m")
    print(f"Frequency range: {scenario.f_c_min/1e9:.0f} - {scenario.f_c_max/1e9:.0f} GHz")