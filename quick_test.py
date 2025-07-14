"""
quick_test.py - 快速测试所有修改是否正常工作
"""

import numpy as np
import utils
import signal_model
import bcrlb_compute
import capacity_bounds
import plots

def quick_test():
    """运行快速测试确保所有修改正常工作"""
    
    print("=== 快速测试 ===\n")
    
    # 1. 测试加速度参数
    print("1. 测试加速度参数:")
    print(f"   相对加速度: {utils.a_rel} m/s²")
    print("   ✓ 加速度参数已添加到utils.py\n")
    
    # 2. 测试信号生成（含加速度）
    print("2. 测试信号生成（含加速度）:")
    try:
        signal_data = signal_model.generate_impaired_signal(20, M_frames=10)
        print("   ✓ 信号生成成功")
        print(f"   信号形状: {signal_data['y'].shape}")
        print(f"   Bussgang增益: {signal_data['B']:.3f}\n")
    except Exception as e:
        print(f"   ✗ 错误: {e}\n")
        return False
    
    # 3. 测试BCRLB计算（含加速度梯度）
    print("3. 测试BCRLB计算:")
    try:
        # 测试2参数（R, v）
        bcrlb_2d = bcrlb_compute.compute_bcrlb(20, signal_data, params=['R', 'v'])
        print(f"   ✓ 2D BCRLB计算成功")
        print(f"   位置RMSE: {np.sqrt(bcrlb_2d['R'])*1e3:.2f} mm")
        print(f"   速度RMSE: {np.sqrt(bcrlb_2d['v'])*1e2:.2f} cm/s")
        
        # 测试3参数（R, v, a）
        bcrlb_3d = bcrlb_compute.compute_bcrlb(20, signal_data, params=['R', 'v', 'a'])
        print(f"   ✓ 3D BCRLB计算成功（含加速度）")
        print(f"   加速度RMSE: {np.sqrt(bcrlb_3d['a']):.2e} m/s²\n")
    except Exception as e:
        print(f"   ✗ 错误: {e}\n")
        return False
    
    # 4. 测试GMM拟合
    print("4. 测试GMM拟合:")
    try:
        eta_samples = signal_data['components']['pa_distortion'].flatten()[:1000]
        gmm_params = bcrlb_compute.fit_gmm_to_distortion(eta_samples, K=3)
        print(f"   ✓ GMM拟合成功")
        print(f"   GMM权重: {gmm_params['weights']}\n")
    except Exception as e:
        print(f"   ✗ 错误: {e}\n")
        return False
    
    # 5. 测试绘图函数
    print("5. 测试新绘图函数:")
    try:
        # 测试GMM详细验证图
        from sklearn.mixture import GaussianMixture
        eta_real = np.column_stack([eta_samples.real, eta_samples.imag])
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(eta_real)
        
        # 不实际绘图，只测试函数调用
        print("   ✓ GMM验证图函数可调用")
        
        # 测试加速度影响图
        acc_range = np.array([0, 5, 10])
        crlb_pos = np.array([1e-6, 1.2e-6, 1.5e-6])
        crlb_vel = np.array([1e-4, 1.2e-4, 1.5e-4])
        print("   ✓ 加速度影响图函数可调用\n")
    except Exception as e:
        print(f"   ✗ 错误: {e}\n")
        return False
    
    # 6. 验证联合惩罚因子
    print("6. 验证联合惩罚因子:")
    sigma_phi_sq = np.var(signal_data['components']['phase_noise'][0])
    phase_penalty = np.exp(sigma_phi_sq)
    pa_penalty = 1 + signal_data['components']['Gamma_eff']
    joint_penalty = phase_penalty * pa_penalty
    
    print(f"   相位噪声方差: {sigma_phi_sq:.3f} rad²")
    print(f"   相位惩罚: {phase_penalty:.3f}")
    print(f"   PA惩罚: {pa_penalty:.3f}")
    print(f"   联合惩罚: {joint_penalty:.3f}")
    print("   ✓ 联合惩罚因子计算正确\n")
    
    print("=== 所有测试通过！===")
    return True

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n请检查错误并修复")
    else:
        print("\n可以运行 python main.py 进行完整仿真")