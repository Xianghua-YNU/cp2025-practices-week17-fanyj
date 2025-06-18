#!/usr/bin/env python3
"""
求解正负电荷构成的泊松方程
文件：poisson_equation_solver.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛迭代法求解二维泊松方程
    
    参数:
        M (int): 每边的网格点数，默认100
        target (float): 收敛精度，默认1e-6
        max_iterations (int): 最大迭代次数，默认10000
    
    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布数组，形状为(M+1, M+1)
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛
    """
    # 设置网格间距
    h = 1.0 / M
    
    # 初始化电势数组
    phi = np.zeros((M+1, M+1), dtype=float)
    phi_prev = np.copy(phi)
    
    # 创建电荷密度数组
    rho = np.zeros((M+1, M+1), dtype=float)
    
    # 缩放电荷位置以适应不同网格大小
    pos_y1, pos_y2 = int(0.6*M), int(0.8*M)
    pos_x1, pos_x2 = int(0.2*M), int(0.4*M)
    neg_y1, neg_y2 = int(0.2*M), int(0.4*M)
    neg_x1, neg_x2 = int(0.6*M), int(0.8*M)
    
    # 设置电荷分布
    rho[pos_y1:pos_y2, pos_x1:pos_x2] = 1.0  # 正电荷
    rho[neg_y1:neg_y2, neg_x1:neg_x2] = -1.0  # 负电荷
    
    # 初始化迭代变量
    delta = 1.0
    iterations = 0
    converged = False
    
    # 主迭代循环
    while delta > target and iterations < max_iterations:
        # 使用有限差分公式更新内部网格点
        phi[1:-1, 1:-1] = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] + 
                                 phi[1:-1, :-2] + phi[1:-1, 2:] + 
                                 h**2 * rho[1:-1, 1:-1])
        
        # 计算最大变化量
        delta = np.max(np.abs(phi - phi_prev))
        
        # 更新前一步解
        phi_prev = np.copy(phi)
        
        # 增加迭代计数
        iterations += 1
    
    # 检查是否收敛
    converged = bool(delta <= target)
    
    # 返回结果
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    可视化电势分布
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制电势分布，添加双线性插值使图像更平滑
    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower', cmap='RdBu_r', interpolation='bilinear')
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Electric Potential (V)', fontsize=12)
    
    # 标注电荷位置
    plt.fill_between([20, 40], [60, 60], [80, 80], alpha=0.3, color='red', label='Positive Charge')
    plt.fill_between([60, 80], [20, 20], [40, 40], alpha=0.3, color='blue', label='Negative Charge')
    
    # 添加标签和标题
    plt.xlabel('x (grid points)', fontsize=12)
    plt.ylabel('y (grid points)', fontsize=12)
    plt.title('Electric Potential Distribution\nPoisson Equation with Positive and Negative Charges', fontsize=14)
    plt.legend()
    
    # 添加网格线
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    分析解的统计信息
    """
    # 打印基本信息
    print(f"Solution Analysis:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {converged}")
    print(f"  Max potential: {np.max(phi):.6f} V")
    print(f"  Min potential: {np.min(phi):.6f} V")
    print(f"  Potential range: {np.max(phi) - np.min(phi):.6f} V")
    
    # 找到极值位置
    max_idx = np.unravel_index(np.argmax(phi), phi.shape)
    min_idx = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"  Max potential location: ({max_idx[0]}, {max_idx[1]})")
    print(f"  Min potential location: ({min_idx[0]}, {min_idx[1]})")

if __name__ == "__main__":
    # 测试代码区域
    print("Solving 2D Poisson equation with relaxation method...")
    
    # 设置参数
    M = 100
    target = 1e-6
    max_iter = 10000
    
    # 调用求解函数
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    
    # 分析结果
    analyze_solution(phi, iterations, converged)
    
    # 可视化结果
    visualize_solution(phi, M)
    
    # 附加分析：沿中心线的电势分布
    plt.figure(figsize=(12, 5))
    
    # 水平横截面
    plt.subplot(1, 2, 1)
    center_y = M // 2
    plt.plot(phi[center_y, :], 'b-', linewidth=2)
    plt.xlabel('x (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along y = {center_y}')
    plt.grid(True, alpha=0.3)
    
    # 垂直横截面
    plt.subplot(1, 2, 2)
    center_x = M // 2
    plt.plot(phi[:, center_x], 'r-', linewidth=2)
    plt.xlabel('y (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along x = {center_x}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
