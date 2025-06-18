#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Completed Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import laplace
import time

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using Successive Over-Relaxation (SOR) method
    for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction  
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # Initialize potential grid
    U = np.zeros((ny, nx))
    
    # Create conductor mask
    conductor_mask = np.zeros((ny, nx), dtype=bool)
    
    # Define conductor regions
    # Upper plate: +100V
    conductor_left = nx//4
    conductor_right = nx//4*3
    y_upper_start = ny // 2 + plate_separation // 2
    y_upper_end = y_upper_start + plate_thickness
    conductor_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    U[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0
    
    # Lower plate: -100V
    y_lower_end = ny // 2 - plate_separation // 2
    y_lower_start = y_lower_end - plate_thickness
    conductor_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    U[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0
    
    # Boundary conditions: grounded sides
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[0, :] = 0.0
    U[-1, :] = 0.0
    
    # SOR iteration with checkerboard pattern for faster convergence
    for iteration in range(max_iter):
        max_error = 0.0
        
        # Red-black checkerboard pattern - update red points first
        for i in range(1, ny-1):
            for j in range(1 + (i%2), nx-1, 2):
                if not conductor_mask[i, j]:  # Skip conductor points
                    old_val = U[i, j]
                    U[i, j] = (1 - omega) * old_val + (omega / 4) * (
                        U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1]
                    )
                    max_error = max(max_error, abs(U[i, j] - old_val))
        
        # Update black points
        for i in range(1, ny-1):
            for j in range(1 + ((i+1)%2), nx-1, 2):
                if not conductor_mask[i, j]:  # Skip conductor points
                    old_val = U[i, j]
                    U[i, j] = (1 - omega) * old_val + (omega / 4) * (
                        U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1]
                    )
                    max_error = max(max_error, abs(U[i, j] - old_val))
        
        # Check convergence
        if max_error < tolerance:
            print(f"Converged after {iteration + 1} iterations with max error {max_error:.6e}")
            break
    else:
        print(f"Warning: Maximum iterations ({max_iter}) reached with max error {max_error:.6e}")
    
    return U

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation: rho = -1/(4*pi) * nabla^2(U)
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # Calculate Laplacian using scipy.ndimage.laplace with proper grid spacing
    laplacian_U = laplace(potential_grid, mode='nearest') / (dx**2)
    
    # Charge density from Poisson equation: rho = -1/(4*pi) * nabla^2(U)
    rho = -laplacian_U / (4 * np.pi)
    
    return rho

def plot_results(potential, charge_density, x_coords, y_coords, plate_info):
    """
    Create comprehensive visualization of results
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
        plate_info (dict): Dictionary containing plate dimensions and positions
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    nx, ny = len(x_coords), len(y_coords)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: 3D Visualization of Potential with contour projection
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_wireframe(X, Y, potential, rstride=3, cstride=3, color='blue', alpha=0.5)
    levels = np.linspace(potential.min(), potential.max(), 20)
    ax1.contourf(X, Y, potential, zdir='z', offset=potential.min(), levels=levels, cmap=cm.viridis)
    
    # Plot conductor plates
    conductor_left = plate_info['conductor_left'] * (x_coords[-1] / nx)
    conductor_right = plate_info['conductor_right'] * (x_coords[-1] / nx)
    y_upper_start = plate_info['y_upper_start'] * (y_coords[-1] / ny)
    y_upper_end = plate_info['y_upper_end'] * (y_coords[-1] / ny)
    y_lower_start = plate_info['y_lower_start'] * (y_coords[-1] / ny)
    y_lower_end = plate_info['y_lower_end'] * (y_coords[-1] / ny)
    
    # Create rectangles for plates
    plate1_x = [conductor_left, conductor_right, conductor_right, conductor_left, conductor_left]
    plate1_y = [y_upper_start, y_upper_start, y_upper_end, y_upper_end, y_upper_start]
    plate2_x = [conductor_left, conductor_right, conductor_right, conductor_left, conductor_left]
    plate2_y = [y_lower_start, y_lower_start, y_lower_end, y_lower_end, y_lower_start]
    
    # Plot plates in 3D
    ax1.plot(plate1_x, plate1_y, 100, 'r-', linewidth=2)
    ax1.plot(plate2_x, plate2_y, -100, 'r-', linewidth=2)
    
    ax1.set_title('3D Visualization of Electric Potential')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Potential (V)')
    ax1.view_init(elev=30, azim=45)
    
    # Subplot 2: 2D Contour Plot of Potential
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, potential, levels=50, cmap=cm.viridis)
    cbar = fig.colorbar(contour, ax=ax2, label='Potential (V)')
    
    # Plot conductor plates
    ax2.fill_between([conductor_left, conductor_right], y_upper_start, y_upper_end, color='red', alpha=0.5)
    ax2.fill_between([conductor_left, conductor_right], y_lower_start, y_lower_end, color='blue', alpha=0.5)
    
    ax2.set_title('Electric Potential Contours')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    # Subplot 3: 3D Charge Density Distribution
    ax3 = fig.add_subplot(223, projection='3d')
    surf = ax3.plot_surface(X, Y, charge_density, cmap='RdBu_r', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, label='Charge Density')
    
    # Plot conductor plates
    ax3.plot(plate1_x, plate1_y, 0, 'k-', linewidth=2)
    ax3.plot(plate2_x, plate2_y, 0, 'k-', linewidth=2)
    
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_zlabel('Charge Density')
    ax3.set_title('3D Charge Density Distribution')
    ax3.view_init(elev=30, azim=45)
    
    # Subplot 4: 2D Heatmap of Charge Density
    ax4 = fig.add_subplot(224)
    # Use symmetric color scale
    max_val = np.max(np.abs(charge_density))
    im = ax4.imshow(charge_density, cmap='RdBu_r', origin='lower', 
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   vmin=-max_val, vmax=max_val, aspect='auto')
    fig.colorbar(im, ax=ax4, label='Charge Density')
    
    # Plot conductor plates
    ax4.fill_between([conductor_left, conductor_right], y_upper_start, y_upper_end, color='gray', alpha=0.3)
    ax4.fill_between([conductor_left, conductor_right], y_lower_start, y_lower_end, color='gray', alpha=0.3)
    
    ax4.set_title('Charge Density Heatmap')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    
    plt.tight_layout()
    plt.show()

def calculate_capacitance(charge_density, dx, dy, plate_info):
    """
    Calculate capacitance from charge density and potential difference
    
    Args:
        charge_density (np.ndarray): Charge density distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        plate_info (dict): Dictionary containing plate dimensions and positions
        
    Returns:
        float: Capacitance value
    """
    # Extract plate dimensions
    conductor_left = plate_info['conductor_left']
    conductor_right = plate_info['conductor_right']
    y_upper_start = plate_info['y_upper_start']
    y_upper_end = plate_info['y_upper_end']
    y_lower_start = plate_info['y_lower_start']
    y_lower_end = plate_info['y_lower_end']
    
    # Calculate total charge on upper plate
    upper_plate_mask = np.zeros_like(charge_density, dtype=bool)
    upper_plate_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    Q_upper = np.sum(charge_density[upper_plate_mask]) * dx * dy
    
    # Calculate total charge on lower plate
    lower_plate_mask = np.zeros_like(charge_density, dtype=bool)
    lower_plate_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    Q_lower = np.sum(charge_density[lower_plate_mask]) * dx * dy
    
    # Calculate average charge magnitude
    Q_avg = (abs(Q_upper) + abs(Q_lower)) / 2
    
    # Potential difference between plates
    V = 200.0  # 100V - (-100V)
    
    # Capacitance
    C = Q_avg / V
    
    return C

if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 120, 100  # Grid dimensions
    plate_thickness = 10  # Conductor thickness in grid points
    plate_separation = 40  # Distance between plates
    omega = 1.9  # SOR relaxation factor
    
    # Physical dimensions
    Lx, Ly = 1.0, 1.0  # Domain size (meters)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Create coordinate arrays
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    
    print("Solving finite thickness parallel plate capacitor...")
    print(f"Grid size: {nx} x {ny}")
    print(f"Plate thickness: {plate_thickness} grid points")
    print(f"Plate separation: {plate_separation} grid points")
    print(f"SOR relaxation factor: {omega}")
    
    # Solve Laplace equation
    start_time = time.time()
    potential = solve_laplace_sor(
        nx, ny, plate_thickness, plate_separation, omega
    )
    solve_time = time.time() - start_time
    
    print(f"Solution completed in {solve_time:.2f} seconds")
    
    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Gather plate information for plotting
    plate_info = {
        'conductor_left': nx//4,
        'conductor_right': nx//4*3,
        'y_upper_start': ny // 2 + plate_separation // 2,
        'y_upper_end': ny // 2 + plate_separation // 2 + plate_thickness,
        'y_lower_start': ny // 2 - plate_separation // 2 - plate_thickness,
        'y_lower_end': ny // 2 - plate_separation // 2
    }
    
    # Calculate capacitance
    C = calculate_capacitance(charge_density, dx, dy, plate_info)
    print(f"\nCalculated Capacitance: {C:.10f} F")
    
    # For comparison, theoretical capacitance of parallel plate capacitor
    plate_area = (plate_info['conductor_right'] - plate_info['conductor_left']) * dx * \
                 (plate_info['y_upper_end'] - plate_info['y_upper_start']) * dy
    plate_distance = (plate_info['y_upper_start'] - plate_info['y_lower_end']) * dy
    epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
    C_theoretical = epsilon_0 * plate_area / plate_distance
    print(f"Theoretical Capacitance: {C_theoretical:.10f} F")
    print(f"Percentage Difference: {abs(C - C_theoretical)/C_theoretical * 100:.2f}%")
    
    # Visualize results
    plot_results(potential, charge_density, x_coords, y_coords, plate_info)
    
    # Print some statistics
    print(f"\nPotential statistics:")
    print(f"  Minimum potential: {np.min(potential):.2f} V")
    print(f"  Maximum potential: {np.max(potential):.2f} V")
    print(f"  Potential range: {np.max(potential) - np.min(potential):.2f} V")
    
    print(f"\nCharge density statistics:")
    print(f"  Maximum positive charge density: {np.max(charge_density):.6e}")
    print(f"  Maximum negative charge density: {np.min(charge_density):.6e}")
    print(f"  Total charge: {np.sum(charge_density) * dx * dy:.10e} C (should be ~0)")    
