"""
plots.py - Generate and save visualizations for THz ISL ISAC validation
Creates figures matching paper specifications
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def setup_plot_style():
    """Set consistent plot style matching IEEE papers"""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'grid.alpha': 0.3,
        'axes.grid': True
    })

def plot_crlb_vs_snr(snr_db, bcrlb_data, crlb_awgn_data, param='R', save_dir='results'):
    """
    Plot CRLB vs SNR showing proposed model, AWGN baseline, and floor
    
    Args:
        snr_db: SNR values in dB
        bcrlb_data: BCRLB values for each SNR
        crlb_awgn_data: AWGN CRLB values
        param: Parameter name ('R' or 'v')
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Convert to standard deviation (square root of CRLB)
    if param == 'R':
        bcrlb_std = np.sqrt(bcrlb_data) * 1e3  # Convert to mm
        crlb_std = np.sqrt(crlb_awgn_data) * 1e3
        ylabel = 'Position RMSE [mm]'
        title = 'Position Estimation Performance'
        floor_value = 0.1  # mm
    else:  # velocity
        bcrlb_std = np.sqrt(bcrlb_data) * 1e2  # Convert to cm/s
        crlb_std = np.sqrt(crlb_awgn_data) * 1e2
        ylabel = 'Velocity RMSE [cm/s]'
        title = 'Velocity Estimation Performance'
        floor_value = 1.0  # cm/s
    
    # Plot lines
    ax.semilogy(snr_db, crlb_std, 'b-', label='AWGN Baseline', linewidth=2)
    ax.semilogy(snr_db, bcrlb_std, 'r--', label='Proposed Model', linewidth=2)
    ax.axhline(y=floor_value, color='g', linestyle=':', label='Hardware Floor', linewidth=2)
    
    # Annotations
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([snr_db[0], snr_db[-1]])
    ax.set_ylim([floor_value/10, max(bcrlb_std)*10])
    
    # Add penalty factor annotation at high SNR
    high_snr_idx = -1
    penalty = bcrlb_std[high_snr_idx] / crlb_std[high_snr_idx]
    ax.annotate(f'{penalty:.1f}× penalty', 
                xy=(snr_db[high_snr_idx], bcrlb_std[high_snr_idx]),
                xytext=(snr_db[high_snr_idx]-10, bcrlb_std[high_snr_idx]*3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                fontsize=9)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'crlb_{param.lower()}_vs_snr.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_capacity_vs_snr(snr_db, capacity_hw, capacity_awgn, capacity_ceiling, save_dir='results'):
    """
    Plot capacity vs SNR showing hardware-limited capacity and AWGN baseline
    
    Args:
        snr_db: SNR values in dB
        capacity_hw: Hardware-limited capacity
        capacity_awgn: AWGN capacity
        capacity_ceiling: Saturation level
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot capacity curves
    ax.plot(snr_db, capacity_awgn, 'b-', label='AWGN Channel', linewidth=2)
    ax.plot(snr_db, capacity_hw, 'r--', label='Hardware-Limited', linewidth=2)
    ax.axhline(y=capacity_ceiling, color='g', linestyle=':', 
               label=f'Ceiling: {capacity_ceiling:.2f} bits/symbol', linewidth=2)
    
    # Fill gap area
    ax.fill_between(snr_db, capacity_awgn, capacity_hw, alpha=0.2, color='red',
                    label='Capacity Loss')
    
    # Annotations
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Capacity [bits/symbol]')
    ax.set_title('Communication Capacity with Hardware Impairments')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([snr_db[0], snr_db[-1]])
    ax.set_ylim([0, max(capacity_awgn[-1], capacity_ceiling) * 1.1])
    
    # Add gap annotation
    mid_idx = len(snr_db) // 2
    gap = capacity_awgn[mid_idx] - capacity_hw[mid_idx]
    ax.annotate(f'Gap: {gap:.1f} bits/symbol',
                xy=(snr_db[mid_idx], (capacity_awgn[mid_idx] + capacity_hw[mid_idx])/2),
                xytext=(snr_db[mid_idx]+5, (capacity_awgn[mid_idx] + capacity_hw[mid_idx])/2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=9)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'capacity_vs_snr.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_capacity_sensitivity(snr_db, capacity_matrix, param_values, param_name, save_dir='results'):
    """
    Plot capacity sensitivity to hardware parameters
    
    Args:
        snr_db: SNR values
        capacity_matrix: 2D capacity values [param x SNR]
        param_values: Parameter values
        param_name: 'phase_noise' or 'distortion'
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot for each parameter value
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    for i, (param_val, color) in enumerate(zip(param_values, colors)):
        if param_name == 'phase_noise':
            label = f'σ²_φ = {param_val:.2f}'
        else:
            label = f'Γ_eff = {param_val:.2f}'
        
        ax.plot(snr_db, capacity_matrix[i, :], color=color, label=label, linewidth=2)
    
    # Annotations
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Capacity [bits/symbol]')
    
    if param_name == 'phase_noise':
        ax.set_title('Capacity Sensitivity to Phase Noise')
    else:
        ax.set_title('Capacity Sensitivity to Hardware Distortion')
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([snr_db[0], snr_db[-1]])
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'capacity_sensitivity_{param_name}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_gmm_validation(empirical_variance, bcrlb_variance, snr_db, save_dir='results'):
    """
    Plot Monte Carlo validation of BCRLB showing GMM approximation accuracy
    
    Args:
        empirical_variance: MC empirical variances
        bcrlb_variance: BCRLB theoretical variances
        snr_db: SNR values
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Compute ratio
    ratio = empirical_variance / bcrlb_variance
    
    # Plot ratio vs SNR
    ax.plot(snr_db, ratio, 'ko-', label='Empirical/BCRLB Ratio', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect Match', linewidth=2)
    ax.fill_between(snr_db, 0.9, 1.1, alpha=0.2, color='green', label='±10% Band')
    
    # Annotations
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Empirical Variance / BCRLB')
    ax.set_title('GMM-Based BCRLB Validation')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([snr_db[0], snr_db[-1]])
    ax.set_ylim([0.5, 1.5])
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'gmm_validation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_gmm_validation_detailed(noise_samples, fitted_gmm, save_dir='results'):
    """
    Plot GMM fitting validation - histogram vs fitted PDF
    验证GMM近似非高斯噪声的准确性
    
    Args:
        noise_samples: Complex noise samples (PA distortion)
        fitted_gmm: Fitted sklearn GMM object
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert complex to 2D real
    noise_real = np.column_stack([noise_samples.real, noise_samples.imag])
    
    # Left plot: Real part
    ax1.hist(noise_samples.real, bins=50, density=True, alpha=0.7, 
             color='blue', edgecolor='black', label='Empirical')
    
    # Generate fine grid for PDF
    x_range = np.linspace(noise_samples.real.min(), noise_samples.real.max(), 200)
    
    # Compute GMM PDF for real part (marginalized)
    # For 2D GMM, we need to integrate over imaginary part
    pdf_real = np.zeros_like(x_range)
    for i, x in enumerate(x_range):
        # Create grid points
        y_range = np.linspace(noise_samples.imag.min(), noise_samples.imag.max(), 50)
        points = np.column_stack([np.full_like(y_range, x), y_range])
        # Compute PDF and marginalize
        pdf_2d = np.exp(fitted_gmm.score_samples(points))
        pdf_real[i] = np.trapz(pdf_2d, y_range)
    
    ax1.plot(x_range, pdf_real, 'r-', linewidth=2, label=f'GMM (K={fitted_gmm.n_components})')
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('PA Distortion - Real Component')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Imaginary part
    ax2.hist(noise_samples.imag, bins=50, density=True, alpha=0.7, 
             color='green', edgecolor='black', label='Empirical')
    
    # Compute GMM PDF for imaginary part
    y_range = np.linspace(noise_samples.imag.min(), noise_samples.imag.max(), 200)
    pdf_imag = np.zeros_like(y_range)
    for i, y in enumerate(y_range):
        x_range_temp = np.linspace(noise_samples.real.min(), noise_samples.real.max(), 50)
        points = np.column_stack([x_range_temp, np.full_like(x_range_temp, y)])
        pdf_2d = np.exp(fitted_gmm.score_samples(points))
        pdf_imag[i] = np.trapz(pdf_2d, x_range_temp)
    
    ax2.plot(y_range, pdf_imag, 'r-', linewidth=2, label=f'GMM (K={fitted_gmm.n_components})')
    ax2.set_xlabel('Imaginary Part')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('PA Distortion - Imaginary Component')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add KL divergence or other goodness-of-fit metric
    # Compute empirical vs GMM KL divergence (approximate)
    hist_real, bin_edges_real = np.histogram(noise_samples.real, bins=50, density=True)
    bin_centers_real = (bin_edges_real[:-1] + bin_edges_real[1:]) / 2
    
    # Avoid log(0) issues
    hist_real[hist_real == 0] = 1e-10
    
    # Compute GMM PDF at bin centers
    gmm_pdf_at_bins = np.zeros_like(bin_centers_real)
    for i, x in enumerate(bin_centers_real):
        points = np.column_stack([np.full(50, x), 
                                 np.linspace(noise_samples.imag.min(), 
                                           noise_samples.imag.max(), 50)])
        pdf_2d = np.exp(fitted_gmm.score_samples(points))
        gmm_pdf_at_bins[i] = np.mean(pdf_2d)
    
    gmm_pdf_at_bins[gmm_pdf_at_bins == 0] = 1e-10
    
    # KL divergence approximation
    kl_div = np.sum(hist_real * np.log(hist_real / gmm_pdf_at_bins)) * (bin_edges_real[1] - bin_edges_real[0])
    
    # 计算相对KL散度（百分比）
    kl_div_percent = 100 * kl_div
    
    fig.suptitle(f'GMM Validation for PA Distortion (KL Divergence: {kl_div:.4f}, {kl_div_percent:.1f}%)', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'gmm_validation_detailed.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_crlb_vs_acceleration(accelerations, crlb_position, crlb_velocity, save_dir='results'):
    """
    Plot CRLB vs acceleration - 展示加速度对估计精度的影响
    
    Args:
        accelerations: Array of acceleration values [m/s^2]
        crlb_position: Position CRLB for each acceleration
        crlb_velocity: Velocity CRLB for each acceleration
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Position CRLB vs acceleration
    ax1.semilogy(accelerations, np.sqrt(crlb_position) * 1e3, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Relative Acceleration [m/s²]')
    ax1.set_ylabel('Position RMSE [mm]')
    ax1.set_title('Position Estimation vs Acceleration')
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for typical LEO range
    ax1.axvspan(0, 10, alpha=0.2, color='green', label='Typical LEO Range')
    ax1.legend()
    
    # Velocity CRLB vs acceleration
    ax2.semilogy(accelerations, np.sqrt(crlb_velocity) * 1e2, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Relative Acceleration [m/s²]')
    ax2.set_ylabel('Velocity RMSE [cm/s]')
    ax2.set_title('Velocity Estimation vs Acceleration')
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(0, 10, alpha=0.2, color='green', label='Typical LEO Range')
    ax2.legend()
    
    fig.suptitle('Impact of Relative Acceleration on Estimation Performance', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'crlb_vs_acceleration.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    """
    Plot Monte Carlo validation of BCRLB showing GMM approximation accuracy
    
    Args:
        empirical_variance: MC empirical variances
        bcrlb_variance: BCRLB theoretical variances
        snr_db: SNR values
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Compute ratio
    ratio = empirical_variance / bcrlb_variance
    
    # Plot ratio vs SNR
    ax.plot(snr_db, ratio, 'ko-', label='Empirical/BCRLB Ratio', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect Match', linewidth=2)
    ax.fill_between(snr_db, 0.9, 1.1, alpha=0.2, color='green', label='±10% Band')
    
    # Annotations
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Empirical Variance / BCRLB')
    ax.set_title('GMM-Based BCRLB Validation')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([snr_db[0], snr_db[-1]])
    ax.set_ylim([0.5, 1.5])
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'gmm_validation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_dse_impact(crlb_pre_dse, crlb_post_dse, snr_db, save_dir='results'):
    """
    Plot CRLB before and after DSE compensation
    
    Args:
        crlb_pre_dse: CRLB without DSE compensation
        crlb_post_dse: CRLB with DSE compensation
        snr_db: SNR values
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Convert to standard deviation in mm
    std_pre = np.sqrt(crlb_pre_dse) * 1e3
    std_post = np.sqrt(crlb_post_dse) * 1e3
    
    # Plot
    ax.semilogy(snr_db, std_pre, 'r-', label='Without DSE Compensation', linewidth=2)
    ax.semilogy(snr_db, std_post, 'b--', label='With DSE Compensation', linewidth=2)
    
    # Add improvement factor
    improvement = std_pre / std_post
    ax2 = ax.twinx()
    ax2.plot(snr_db, improvement, 'g:', label='Improvement Factor', linewidth=2)
    ax2.set_ylabel('Improvement Factor', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Annotations
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Position RMSE [mm]')
    ax.set_title('Impact of DSE Compensation on Estimation Accuracy')
    ax.legend(loc='upper right')
    ax2.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'dse_impact.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def create_summary_figure(results_dict, save_dir='results'):
    """
    Create a multi-panel summary figure
    
    Args:
        results_dict: Dictionary containing all results
        save_dir: Directory to save plot
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(12, 8))
    
    # Panel 1: Position CRLB
    ax1 = plt.subplot(2, 2, 1)
    snr_db = results_dict['snr_db']
    bcrlb_r = np.sqrt(results_dict['bcrlb_position']) * 1e3
    crlb_r = np.sqrt(results_dict['crlb_awgn_position']) * 1e3
    
    ax1.semilogy(snr_db, crlb_r, 'b-', label='AWGN')
    ax1.semilogy(snr_db, bcrlb_r, 'r--', label='Hardware-Limited')
    ax1.set_xlabel('SNR [dB]')
    ax1.set_ylabel('Position RMSE [mm]')
    ax1.set_title('(a) Position Estimation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Velocity CRLB
    ax2 = plt.subplot(2, 2, 2)
    bcrlb_v = np.sqrt(results_dict['bcrlb_velocity']) * 1e2
    crlb_v = np.sqrt(results_dict['crlb_awgn_velocity']) * 1e2
    
    ax2.semilogy(snr_db, crlb_v, 'b-', label='AWGN')
    ax2.semilogy(snr_db, bcrlb_v, 'r--', label='Hardware-Limited')
    ax2.set_xlabel('SNR [dB]')
    ax2.set_ylabel('Velocity RMSE [cm/s]')
    ax2.set_title('(b) Velocity Estimation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Capacity
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(snr_db, results_dict['capacity_awgn'], 'b-', label='AWGN')
    ax3.plot(snr_db, results_dict['capacity_hw'], 'r--', label='Hardware-Limited')
    ax3.axhline(y=results_dict['capacity_ceiling'], color='g', linestyle=':', 
                label=f'Ceiling: {results_dict["capacity_ceiling"]:.2f}')
    ax3.set_xlabel('SNR [dB]')
    ax3.set_ylabel('Capacity [bits/symbol]')
    ax3.set_title('(c) Communication Capacity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""System Parameters:
• Carrier: {results_dict['f_c']/1e9:.0f} GHz
• Bandwidth: {results_dict['B']/1e9:.0f} GHz
• Range: {results_dict['R_0']/1e3:.0f} km
• Velocity: {results_dict['v_rel']/1e3:.0f} km/s

Key Results (at 30 dB SNR):
• Position penalty: {results_dict['position_penalty_30dB']:.1f}×
• Velocity penalty: {results_dict['velocity_penalty_30dB']:.1f}×
• Capacity gap: {results_dict['capacity_gap_30dB']:.1f} bits/symbol
• Capacity ceiling: {results_dict['capacity_ceiling']:.2f} bits/symbol"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('(d) Summary')
    
    plt.suptitle('THz ISL ISAC Performance Analysis', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'summary_figure.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")