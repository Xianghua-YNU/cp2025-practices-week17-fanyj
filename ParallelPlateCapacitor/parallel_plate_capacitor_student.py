import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    Solve Laplace equation using Jacobi iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        tol (float): Convergence tolerance
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize potential grid
    u = np.zeros((ygrid, xgrid))
    
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    iterations = 0
    max_iter = 10000
    convergence_history = []
    
    while iterations < max_iter:
        u_old = u.copy()
        
        # Jacobi iteration
        u[1:-1,1:-1] = 0.25*(u[2:,1:-1] + u[:-2,1:-1] + u[1:-1, 2:] + u[1:-1,:-2]) 

        # Maintain boundary conditions
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # Calculate convergence metric
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)

        # Check convergence
        iterations += 1
        if max_change < tol:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    Solve Laplace equation using Gauss-Seidel SOR iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        omega (float): Relaxation factor
        Niter (int): Maximum number of iterations
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize potential grid
    u = np.zeros((ygrid, xgrid))
    
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    convergence_history = []
    
    for iteration in range(Niter):
        u_old = u.copy()
        
        # SOR iteration
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # Skip plate regions
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # Calculate residual
                r_ij = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                
                # Apply SOR formula
                u[i, j] = (1 - omega) * u[i, j] + omega * r_ij
        
        # Maintain boundary conditions
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # Calculate convergence metric
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # Check convergence
        if max_change < tol:
            break
    
    return u, iteration + 1, convergence_history

def plot_results(x, y, u, method_name):
    """
    Plot 3D potential distribution and equipotential contours.
    
    Args:
        x (array): X coordinates
        y (array): Y coordinates
        u (array): Potential distribution
        method_name (str): Name of the method used
    """
    fig = plt.figure(figsize=(15, 6))
    
    # 3D wireframe plot with contour projection
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax1.plot_wireframe(X, Y, u, alpha=0.7, color='blue')
    
    # Add contour projection on z-axis
    levels = np.linspace(u.min(), u.max(), 20)
    ax1.contourf(X, Y, u, zdir='z', offset=u.min(), levels=levels, cmap='viridis', alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential (V)')
    ax1.set_title(f'3D Potential Distribution - {method_name}')
    
    # Equipotential contour plot and Electric field streamlines
    ax2 = fig.add_subplot(122)
    
    # Plot equipotential lines
    contour = ax2.contour(X, Y, u, levels=20, colors='red', linestyles='dashed', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
    
    # Calculate electric field (negative gradient of potential)
    EY, EX = np.gradient(-u, 1)  # Electric field is negative gradient
    
    # Plot electric field lines using streamplot
    ax2.streamplot(X, Y, EX, EY, density=1.5, color='blue', linewidth=1, arrowsize=1.5, arrowstyle='->')
    
    # Highlight the capacitor plates
    ax2.hlines(yB, xL, xR, colors='black', linewidth=2)
    ax2.hlines(yT, xL, xR, colors='black', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Equipotential Lines & Electric Field Lines - {method_name}')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    xgrid, ygrid = 50, 50
    w, d = 20, 20  # plate width and separation
    tol = 1e-3
    
    # Create coordinate arrays
    x = np.linspace(0, xgrid-1, xgrid)
    y = np.linspace(0, ygrid-1, ygrid)
    
    print("Solving Laplace equation for parallel plate capacitor...")
    print(f"Grid size: {xgrid} x {ygrid}")
    print(f"Plate width: {w}, separation: {d}")
    print(f"Tolerance: {tol}")
    
    # Solve using Jacobi method
    print("1. Jacobi iteration method:")
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol=tol)
    time_jacobi = time.time() - start_time
    print(f"   Converged in {iter_jacobi} iterations")
    print(f"   Time: {time_jacobi:.3f} seconds")
    
    # Solve using SOR method with optimal omega for 2D square grid (omega ~ 2/(1+pi/N))
    optimal_omega = 2 / (1 + np.pi/xgrid)
    print(f"\n2. Gauss-Seidel SOR iteration method (ω = {optimal_omega:.3f}):")
    start_time = time.time()
    u_sor, iter_sor, conv_history_sor = solve_laplace_sor(xgrid, ygrid, w, d, omega=optimal_omega, tol=tol)
    time_sor = time.time() - start_time
    print(f"   Converged in {iter_sor} iterations")
    print(f"   Time: {time_sor:.3f} seconds")
    
    # Performance comparison
    print("\n3. Performance comparison:")
    print(f"   Jacobi: {iter_jacobi} iterations, {time_jacobi:.3f}s")
    print(f"   SOR:    {iter_sor} iterations, {time_sor:.3f}s")
    print(f"   Speedup: {iter_jacobi/iter_sor:.1f}x iterations, {time_jacobi/time_sor:.2f}x time")
    
    # Plot results
    plot_results(x, y, u_jacobi, "Jacobi Method")
    plot_results(x, y, u_sor, f"SOR Method (ω = {optimal_omega:.3f})")
    
    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(conv_history_jacobi)), conv_history_jacobi, 'r-', label='Jacobi Method')
    plt.semilogy(range(len(conv_history_sor)), conv_history_sor, 'b-', label=f'SOR Method (ω = {optimal_omega:.3f})')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Change (log scale)')
    plt.title('Convergence Comparison')
    plt.grid(True)
    plt.legend()
    plt.show()
