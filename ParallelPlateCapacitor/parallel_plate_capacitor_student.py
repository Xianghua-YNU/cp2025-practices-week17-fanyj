import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    """
    # 初始化电势网格
    V = np.zeros((ygrid, xgrid))
    
    # 设置边界条件 - 平行板电容器
    plate_start_x = (xgrid - w) // 2
    plate_end_x = plate_start_x + w
    plate_mid_y = ygrid // 2
    
    V[plate_mid_y - d//2, plate_start_x:plate_end_x] = 1.0  # 上极板
    V[plate_mid_y + d//2, plate_start_x:plate_end_x] = -1.0  # 下极板
    
    # 边界条件：所有边界上的电势保持为0（除了极板）
    V[:, 0] = 0
    V[:, -1] = 0
    V[0, :] = 0
    V[-1, :] = 0
    
    # 初始化收敛历史
    convergence_history = []
    iterations = 0
    
    # 创建用于更新的临时数组
    V_new = V.copy()
    
    # 迭代求解
    while True:
        max_diff = 0
        # 内部点更新
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过极板位置
                if (i == plate_mid_y - d//2 or i == plate_mid_y + d//2) and (j >= plate_start_x and j < plate_end_x):
                    continue
                # Jacobi迭代更新
                V_new[i, j] = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
                # 计算最大变化
                diff = abs(V_new[i, j] - V[i, j])
                if diff > max_diff:
                    max_diff = diff
        
        # 更新收敛历史
        convergence_history.append(max_diff)
        iterations += 1
        
        # 检查收敛
        if max_diff < tol:
            break
        
        # 更新V为V_new
        V = V_new.copy()
    
    return V, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    """
    # 初始化电势网格
    V = np.zeros((ygrid, xgrid))
    
    # 设置边界条件 - 平行板电容器
    plate_start_x = (xgrid - w) // 2
    plate_end_x = plate_start_x + w
    plate_mid_y = ygrid // 2
    
    V[plate_mid_y - d//2, plate_start_x:plate_end_x] = 1.0  # 上极板
    V[plate_mid_y + d//2, plate_start_x:plate_end_x] = -1.0  # 下极板
    
    # 边界条件：所有边界上的电势保持为0（除了极板）
    V[:, 0] = 0
    V[:, -1] = 0
    V[0, :] = 0
    V[-1, :] = 0
    
    # 初始化收敛历史
    convergence_history = []
    iterations = 0
    
    # 迭代求解
    for iter in range(Niter):
        max_diff = 0
        # 内部点更新
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过极板位置
                if (i == plate_mid_y - d//2 or i == plate_mid_y + d//2) and (j >= plate_start_x and j < plate_end_x):
                    continue
                # 保存旧值用于计算变化
                old_value = V[i, j]
                # SOR迭代更新
                V[i, j] = (1 - omega) * old_value + omega * 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
                # 计算最大变化
                diff = abs(V[i, j] - old_value)
                if diff > max_diff:
                    max_diff = diff
        
        # 更新收敛历史
        convergence_history.append(max_diff)
        iterations += 1
        
        # 检查收敛
        if max_diff < tol:
            break
    
    return V, iterations, convergence_history

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    """
    # 创建图形和子图
    fig = plt.figure(figsize=(15, 6))
    
    # 计算电场 (E = -∇V)
    Ey, Ex = np.gradient(-u)
    
    # 第一个子图：3D电势分布
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_wireframe(X, Y, u, rstride=1, cstride=1, color='blue', alpha=0.7)
    
    # 在z=0平面上投影等势线
    contour_proj = ax1.contourf(X, Y, u, zdir='z', offset=np.min(u), cmap='viridis', alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('电势 V')
    ax1.set_title(f'{method_name} - 三维电势分布')
    fig.colorbar(contour_proj, ax=ax1, shrink=0.5, aspect=5, label='电势')
    
    # 第二个子图：等势线和电场线
    ax2 = fig.add_subplot(122)
    
    # 绘制等势线
    contour = ax2.contour(X, Y, u, 20, colors='blue', alpha=0.7)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # 绘制电场线（流线图）
    streamplot = ax2.streamplot(X, Y, Ex, Ey, density=1.5, color='red', linewidth=1, arrowsize=1.5)
    
    # 标记极板位置
    plate_start_x = (len(x) - len(x)//4) // 2
    plate_end_x = plate_start_x + len(x)//4
    plate_mid_y = len(y) // 2
    plate_d = len(y)//8
    
    ax2.hlines(y[plate_mid_y - plate_d//2], x[plate_start_x], x[plate_end_x], colors='black', linewidth=2)
    ax2.hlines(y[plate_mid_y + plate_d//2], x[plate_start_x], x[plate_end_x], colors='black', linewidth=2)
    ax2.text(x[plate_start_x], y[plate_mid_y - plate_d//2], 'V=1', va='bottom')
    ax2.text(x[plate_start_x], y[plate_mid_y + plate_d//2], 'V=-1', va='top')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{method_name} - 等势线(蓝)和电场线(红)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置参数
    xgrid = 100  # x方向网格点数
    ygrid = 100  # y方向网格点数
    w = 50       # 平行板宽度
    d = 20       # 平行板间距
    
    # 测试Jacobi方法
    start_time = time.time()
    V_jacobi, iter_jacobi, history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d)
    end_time = time.time()
    print(f"Jacobi方法: 迭代次数 = {iter_jacobi}, 耗时 = {end_time - start_time:.4f}秒")
    
    # 测试SOR方法
    omega = 1.8  # 松弛因子，对于正方形网格，最佳值约为2/(1+sin(pi/N))
    start_time = time.time()
    V_sor, iter_sor, history_sor = solve_laplace_sor(xgrid, ygrid, w, d, omega=omega)
    end_time = time.time()
    print(f"SOR方法 (ω={omega}): 迭代次数 = {iter_sor}, 耗时 = {end_time - start_time:.4f}秒")
    
    # 创建坐标网格
    x = np.linspace(0, 1, xgrid)
    y = np.linspace(0, 1, ygrid)
    
    # 绘制结果
    plot_results(x, y, V_jacobi, "Jacobi方法")
    plot_results(x, y, V_sor, f"SOR方法 (ω={omega})")
    
    # 绘制收敛历史对比图
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, iter_jacobi+1), history_jacobi, label='Jacobi方法')
    plt.semilogy(range(1, iter_sor+1), history_sor, label=f'SOR方法 (ω={omega})')
    plt.xlabel('迭代次数')
    plt.ylabel('最大电势变化 (对数刻度)')
    plt.title('收敛历史对比')
    plt.grid(True)
    plt.legend()
    plt.show()
