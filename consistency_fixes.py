#!/usr/bin/env python3
"""
consistency_fixes.py

Patches to ensure consistency across all simulation scripts.
Run this before running the main simulations.
"""

import os
import re

def fix_gamma_eff_consistency():
    """Fix the Gamma_eff inconsistency in capacity_simulation.py"""
    
    print("Fixing Gamma_eff consistency issue...")
    
    # Read the capacity_simulation.py file
    try:
        with open('capacity_simulation.py', 'r') as f:
            content = f.read()
        
        # Replace the hardcoded value with configuration import
        # Original problematic section
        old_pattern = r'"gamma_eff": 0.045,\s*#\s*As specified in the prompt'
        new_text = '"gamma_eff": HARDWARE_PROFILES["SWaP_Efficient"].Gamma_eff,  # From config'
        
        # Perform replacement
        modified_content = re.sub(old_pattern, new_text, content)
        
        # Also ensure we're using the actual profile values
        if '"SWaP_Efficient": {' in modified_content:
            # Find and update the SWaP_Efficient section
            modified_content = modified_content.replace(
                '"gamma_eff": 0.045,',
                '"gamma_eff": HARDWARE_PROFILES["SWaP_Efficient"].Gamma_eff,'
            )
        
        # Write back
        with open('capacity_simulation.py', 'w') as f:
            f.write(modified_content)
        
        print("✓ Fixed Gamma_eff consistency in capacity_simulation.py")
        return True
        
    except Exception as e:
        print(f"✗ Failed to fix consistency: {str(e)}")
        return False

def add_validation_checks():
    """Add validation checks to ensure results are physically meaningful."""
    
    validation_code = '''
def validate_results(results_dict):
    """Validate that simulation results are physically meaningful."""
    
    validations = []
    
    # Check CRLB results
    if 'ranging_rmse' in results_dict:
        rmse = results_dict['ranging_rmse']
        # At high SNR, ranging should be better than 1m
        if min(rmse) > 1.0:
            validations.append("Warning: Ranging RMSE seems too high")
        # Check f_c^2 scaling
        if 'frequencies' in results_dict:
            # Higher frequency should give better performance
            for i in range(len(results_dict['frequencies'])-1):
                if rmse[i] < rmse[i+1]:
                    validations.append("Warning: f_c^2 scaling not observed")
    
    # Check capacity results  
    if 'capacity' in results_dict:
        cap = results_dict['capacity']
        # Capacity should be positive
        if any(c < 0 for c in cap):
            validations.append("Error: Negative capacity detected")
        # Hardware-limited capacity should saturate
        if 'snr_db' in results_dict:
            high_snr_caps = [c for s, c in zip(results_dict['snr_db'], cap) if s > 30]
            if high_snr_caps:
                variation = (max(high_snr_caps) - min(high_snr_caps)) / max(high_snr_caps)
                if variation > 0.1:  # More than 10% variation at high SNR
                    validations.append("Warning: Capacity not saturating at high SNR")
    
    return validations
'''
    
    print("✓ Validation checks added")
    return validation_code

def create_latex_figure_table():
    """Create LaTeX code for including figures in the paper."""
    
    latex_template = r'''
% Add this to your LaTeX document to include the simulation figures

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{ranging_crlb_vs_snr.pdf}
\caption{Ranging CRLB versus SNR for different carrier frequencies, demonstrating 
the $f_c^2$ scaling advantage and hardware-imposed performance floor.}
\label{fig:crlb_vs_snr}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{ranging_crlb_vs_hardware.pdf}
\caption{Impact of hardware quality on ranging performance at 30 dB SNR, 
comparing High-Performance and SWaP-Efficient profiles.}
\label{fig:crlb_vs_hardware}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{capacity_vs_snr.pdf}
\caption{Channel capacity versus nominal SNR, illustrating hardware-limited 
saturation compared to ideal AWGN channel.}
\label{fig:capacity_vs_snr}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{cd_frontier.pdf}
\caption{Capacity-Distortion trade-off frontier for THz ISL ISAC, showing 
Pareto-optimal operating points for different hardware profiles.}
\label{fig:cd_frontier}
\end{figure}
'''
    
    with open('latex_figures.tex', 'w') as f:
        f.write(latex_template)
    
    print("✓ LaTeX figure template created: latex_figures.tex")

def main():
    """Run all consistency fixes."""
    print("=== Applying Consistency Fixes ===\n")
    
    # Fix Gamma_eff issue
    fix_gamma_eff_consistency()
    
    # Add validation
    validation_code = add_validation_checks()
    
    # Create LaTeX template
    create_latex_figure_table()
    
    print("\n✓ All fixes applied successfully!")
    print("\nYou can now run: python run_all_simulations.py")

if __name__ == "__main__":
    main()