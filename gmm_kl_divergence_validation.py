"""
gmm_kl_divergence_validation.py - 验证GMM近似的KL散度是否在1%以内
支持论文中的声明："maintaining fidelity within 1% KL divergence"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import signal_model
import utils

def compute_kl_divergence_accurate(samples, gmm_model, n_points=1000):
    """
    更精确地计算经验分布与GMM之间的KL散度
    
    Args:
        samples: 复数噪声样本
        gmm_model: 拟合的GMM模型
        n_points: 用于数值积分的网格点数
        
    Returns:
        kl_div: KL散度值
        kl_percent: KL散度百分比
    """
    # 转换为2D实数表示
    samples_2d = np.column_stack([samples.real, samples.imag])
    
    # 使用KDE估计经验分布
    kde = gaussian_kde(samples_2d.T)
    
    # 创建2D网格
    x_min, x_max = samples.real.min(), samples.real.max()
    y_min, y_max = samples.imag.min(), samples.imag.max()
    
    x = np.linspace(x_min, x_max, int(np.sqrt(n_points)))
    y = np.linspace(y_min, y_max, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    
    # 评估点
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # 计算经验PDF (KDE)
    p_empirical = kde(positions).reshape(X.shape)
    
    # 计算GMM PDF
    points = np.column_stack([X.ravel(), Y.ravel()])
    log_p_gmm = gmm_model.score_samples(points)
    p_gmm = np.exp(log_p_gmm).reshape(X.shape)
    
    # 归一化
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    p_empirical = p_empirical / (p_empirical.sum() * dx * dy)
    p_gmm = p_gmm / (p_gmm.sum() * dx * dy)
    
    # 计算KL散度: D(p||q) = ∫ p(x) log(p(x)/q(x)) dx
    # 避免log(0)
    mask = (p_empirical > 1e-10) & (p_gmm > 1e-10)
    kl_div = np.sum(p_empirical[mask] * np.log(p_empirical[mask] / p_gmm[mask])) * dx * dy
    
    # 转换为百分比（相对熵）
    # 归一化因子：使用经验分布的熵
    h_empirical = -np.sum(p_empirical[mask] * np.log(p_empirical[mask])) * dx * dy
    kl_percent = 100 * kl_div / h_empirical if h_empirical > 0 else 0
    
    return kl_div, kl_percent

def validate_gmm_across_snr():
    """
    在不同SNR下验证GMM拟合质量
    """
    snr_values = [10, 20, 30, 40]
    k_values = [2, 3, 4, 5]
    
    results = np.zeros((len(snr_values), len(k_values)))
    
    print("=== GMM KL散度验证 ===\n")
    print("验证GMM近似是否满足 '1% KL divergence' 要求\n")
    
    for i, snr in enumerate(snr_values):
        print(f"SNR = {snr} dB:")
        
        # 生成PA失真样本
        signal_data = signal_model.generate_impaired_signal(snr, M_frames=100)
        eta_samples = signal_data['components']['pa_distortion'].flatten()[:10000]
        eta_2d = np.column_stack([eta_samples.real, eta_samples.imag])
        
        for j, k in enumerate(k_values):
            # 拟合GMM
            gmm = GaussianMixture(n_components=k, covariance_type='full', 
                                 random_state=42, n_init=10)
            gmm.fit(eta_2d)
            
            # 计算KL散度
            kl_div, kl_percent = compute_kl_divergence_accurate(eta_samples, gmm)
            results[i, j] = kl_percent
            
            status = "✓" if kl_percent < 1.0 else "✗"
            print(f"  K={k}: KL = {kl_percent:.2f}% {status}")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    
    for j, k in enumerate(k_values):
        plt.plot(snr_values, results[:, j], 'o-', label=f'K={k}', 
                linewidth=2, markersize=8)
    
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, 
                label='1% Threshold')
    plt.xlabel('SNR [dB]')
    plt.ylabel('KL Divergence [%]')
    plt.title('GMM Approximation Quality vs SNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, max(2.0, results.max() * 1.1)])
    
    plt.tight_layout()
    plt.savefig('results/gmm_kl_divergence_validation.png', dpi=300)
    plt.close()
    
    # 分析结果
    print("\n=== 分析结果 ===")
    
    # 找出满足1%要求的最小K值
    min_k_required = None
    for j, k in enumerate(k_values):
        if np.all(results[:, j] < 1.0):
            min_k_required = k
            break
    
    if min_k_required:
        print(f"最小K值要求: K = {min_k_required}")
    else:
        print("警告：没有K值能在所有SNR下满足1%要求")
    
    # K=3的性能
    k3_idx = k_values.index(3)
    k3_performance = results[:, k3_idx]
    print(f"\nK=3性能总结:")
    print(f"  平均KL散度: {np.mean(k3_performance):.2f}%")
    print(f"  最大KL散度: {np.max(k3_performance):.2f}%")
    print(f"  是否满足1%要求: {'是' if np.all(k3_performance < 1.0) else '否'}")
    
    return results

def plot_gmm_components_physical_meaning():
    """
    可视化GMM三个组件的物理意义
    """
    # 生成高SNR下的PA失真（更明显的非线性）
    signal_data = signal_model.generate_impaired_signal(30, M_frames=100)
    eta_samples = signal_data['components']['pa_distortion'].flatten()[:10000]
    eta_2d = np.column_stack([eta_samples.real, eta_samples.imag])
    
    # 拟合K=3 GMM
    gmm = GaussianMixture(n_components=3, covariance_type='full', 
                         random_state=42, n_init=10)
    gmm.fit(eta_2d)
    
    # 获取组件参数
    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：2D散点图与椭圆
    ax1.scatter(eta_2d[:1000, 0], eta_2d[:1000, 1], alpha=0.3, s=1, c='gray')
    
    # 绘制每个组件的椭圆
    colors = ['blue', 'green', 'red']
    labels = ['Linear (k=1)', 'Compression (k=2)', 'Saturation (k=3)']
    
    for k in range(3):
        # 计算椭圆参数
        mean = means[k]
        cov = covariances[k]
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # 绘制2-sigma椭圆
        from matplotlib.patches import Ellipse
        ell = Ellipse(mean, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]),
                     angle=angle, facecolor=colors[k], alpha=0.3, 
                     edgecolor=colors[k], linewidth=2)
        ax1.add_patch(ell)
        
        # 标记中心
        ax1.plot(mean[0], mean[1], 'o', color=colors[k], markersize=10)
        ax1.text(mean[0], mean[1]+0.02, f'μ{k+1}', fontsize=10, ha='center')
    
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('GMM Components Physical Interpretation')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右图：组件特性表
    ax2.axis('off')
    
    # 创建表格数据
    table_data = []
    table_data.append(['Component', 'Weight', '|μ|', 'tr(Σ)', 'Physical Regime'])
    
    for k in range(3):
        mean_mag = np.sqrt(means[k, 0]**2 + means[k, 1]**2)
        trace_cov = np.trace(covariances[k])
        
        if k == 0:
            regime = 'Linear (Low Power)'
        elif k == 1:
            regime = 'Compression (Medium)'
        else:
            regime = 'Saturation (High Power)'
        
        table_data.append([
            f'k={k+1}',
            f'{weights[k]:.3f}',
            f'{mean_mag:.4f}',
            f'{trace_cov:.4f}',
            regime
        ])
    
    # 绘制表格
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置标题行样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('GMM Component Characteristics', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig('results/gmm_components_physical_meaning.png', dpi=300)
    plt.close()
    
    print("\n组件物理意义总结:")
    for k in range(3):
        print(f"Component {k+1}: λ={weights[k]:.3f}, ", end='')
        if k == 0:
            print("线性区 - 低功率输入，接近高斯分布")
        elif k == 1:
            print("压缩区 - 中等功率，软限幅开始")
        else:
            print("饱和区 - 高功率输入，硬限幅效应")

def main():
    """主函数"""
    print("=" * 60)
    print("GMM近似质量验证 - KL散度分析")
    print("=" * 60)
    
    # 1. 验证不同SNR和K值下的KL散度
    results = validate_gmm_across_snr()
    
    # 2. 可视化GMM组件的物理意义
    print("\n绘制GMM组件物理意义图...")
    plot_gmm_components_physical_meaning()
    
    print("\n验证完成！")
    print("生成的图表:")
    print("- gmm_kl_divergence_validation.png")
    print("- gmm_components_physical_meaning.png")
    
    # 给出建议
    print("\n=== 建议 ===")
    if np.all(results[:, 1] < 1.0):  # K=3的索引是1
        print("✓ K=3满足1% KL散度要求，可以在论文中声明此结果")
    else:
        max_kl = np.max(results[:, 1])
        print(f"⚠ K=3的最大KL散度为{max_kl:.2f}%，建议:")
        print(f"  1. 修改论文声明为 'within {np.ceil(max_kl):.0f}% KL divergence'")
        print("  2. 或考虑使用K=4以满足1%要求")

if __name__ == "__main__":
    main()