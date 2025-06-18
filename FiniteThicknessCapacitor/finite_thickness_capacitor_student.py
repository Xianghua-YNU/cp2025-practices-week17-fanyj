#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Completed Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
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
    # Initialize potential grid with zeros
    potential = np.zeros((ny, nx))
    
    # Calculate plate positions
    plate_center_y = ny // 2
    plate_start_y = plate_center_y - plate_separation // 2
    plate_end_y = plate_center_y + plate_separation // 2
    
    # Set left plate (positive potential, e.g., 1V)
    left_plate_x = nx // 4
    potential[plate_start_y:plate_start_y+plate_thickness, left_plate_x] = 1.0
    
    # Set right plate (negative potential, e.g., -1V)
    right_plate_x = 3 * nx // 4
    potential[plate_end_y-plate_thickness:plate_end_y, right_plate_x] = -1.0
    
    # Create mask for plates (to exclude from SOR updates)
    plate_mask = np.zeros_like(potential, dtype=bool)
    plate_mask[plate_start_y:plate_start_y+plate_thickness, left_plate_x] = True
    plate_mask[plate_end_y-plate_thickness:plate_end_y, right_plate_x] = True
    
    # SOR iteration
    error = tolerance + 1  # Initialize error above tolerance
    iterations = 0
    
    while error > tolerance and iterations < max_iter:
        old_potential = potential.copy()
        
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Skip plate points
                if plate_mask[j, i]:
                    continue
                    
                # SOR update
                potential[j, i] = (1 - omega) * potential[j, i] + \
                                 (omega / 4) * (potential[j, i+1] + potential[j, i-1] + 
                                              potential[j+1, i] + potential[j-1, i])
        
        # Calculate maximum error
        error = np.max(np.abs(potential - old_potential))
        iterations += 1
    
    print(f"Converged in {iterations} iterations with final error {error:.2e}")
    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution (normalized units)
    """
    # Allocate array for charge density
    rho = np.zeros_like(potential_grid)
    
    # Compute Laplacian of potential (finite difference approximation)
    # d^2V/dx^2 + d^2V/dy^2 = -rho/epsilon_0
    # We'll set epsilon_0 = 1 for normalized units
    for j in range(1, potential_grid.shape[0]-1):
        for i in range(1, potential_grid.shape[1]-1):
            d2V_dx2 = (potential_grid[j, i+1] - 2*potential_grid[j, i] + potential_grid[j, i-1]) / (dx*dx)
            d2V_dy2 = (potential_grid[j+1, i] - 2*potential_grid[j, i] + potential_grid[j-1, i]) / (dy*dy)
            rho[j, i] = -(d2V_dx2 + d2V_dy2)
    
    return rho

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 3D Surface plot of electric potential
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, potential, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_title('Electric Potential')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential (V)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot of electric potential
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, potential, 20, cmap=cm.viridis)
    ax2.set_title('Electric Potential Contours')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
    
    # 3D Surface plot of charge density
    ax3 = fig.add_subplot(223, projection='3d')
    surf = ax3.plot_surface(X, Y, charge_density, cmap=cm.plasma, linewidth=0, antialiased=True)
    ax3.set_title('Charge Density')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Charge Density')
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)
    
    # Contour plot of charge density
    ax4 = fig.add_subplot(224)
    # We'll use a symmetric colormap and levels to highlight positive and negative charges
    max_val = np.max(np.abs(charge_density))
    levels = np.linspace(-max_val, max_val, 21)
    contour = ax4.contourf(X, Y, charge_density, levels=levels, cmap=cm.coolwarm)
    ax4.set_title('Charge Density Contours')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    fig.colorbar(contour, ax=ax4, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx = 101  # Number of grid points in x direction
    ny = 101  # Number of grid points in y direction
    plate_thickness = 5  # Thickness of capacitor plates (grid points)
    plate_separation = 40  # Separation between plates (grid points)
    omega = 1.9  # Relaxation factor for SOR method
    max_iterations = 10000  # Maximum iterations
    tolerance = 1e-6  # Convergence tolerance
    
    # Calculate grid spacing (arbitrary units)
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    
    # Solve Laplace equation
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega, max_iterations, tolerance)
    
    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Plot results
    plot_results(potential, charge_density, x_coords, y_coords)
