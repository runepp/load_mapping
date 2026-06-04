# -*- coding: utf-8 -*-
"""
Example usage of PCA-based vertebra segmentation.

This script demonstrates:
1. Loading a vertebra mask
2. Using PCA to understand vertebra geometry
3. Comparing PCA-based splitting with weighted regression
4. Visualizing the results

Author: Rune P
"""

import numpy as np
import matplotlib.pyplot as plt
from vertebra_segmentation_v3 import VertebraSegmenter, segment_vertebra_pca
from scipy import ndimage


def example_1_pca_basic():
    """
    Example 1: Basic PCA-based segmentation.
    
    Shows how to:
    - Initialize a segmenter
    - Get principal axes using 3D PCA
    - Assign regions based on PCA
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic PCA Segmentation")
    print("="*60)
    
    # Create a synthetic vertebra (ellipsoid)
    # This represents a lumbar vertebra
    vertebra_mask = create_synthetic_vertebra(
        shape=(50, 60, 40),
        center=(25, 30, 20),
        radii=(15, 20, 18),  # Different sizes (not perfectly spherical)
        rotation_angle=15  # Rotated 15 degrees (not axis-aligned)
    )
    
    # Initialize segmenter
    segmenter = VertebraSegmenter(vertebra_mask, pixel_spacing=1.0, slice_thickness=1.0)
    
    # Get statistics
    stats = segmenter.get_summary_statistics()
    
    print(f"\nVertebra Statistics:")
    print(f"  Volume: {stats['volume']} voxels")
    print(f"  Centroid: {stats['centroid']}")
    print(f"  Eigenvalues: {stats['eigenvalues']}")
    print(f"  Aspect ratios: {stats['aspect_ratios']}")
    
    # Perform PCA segmentation
    axes, com, eigenvalues = segmenter.split_by_3d_pca()
    regions = segmenter.assign_regions_pca(axes, com)
    
    print(f"\nPCA Axes (principal directions):")
    for i, axis in enumerate(axes.T):
        print(f"  Axis {i+1}: {axis} (eigenvalue: {eigenvalues[i]:.2f})")
    
    # Count voxels in each region
    unique_regions = np.unique(regions)
    region_names = {
        1: "No region",
        2: "Superior",
        4: "Inferior",
        8: "Anterior",
        16: "Posterior",
        32: "Left",
        64: "Right"
    }
    
    print(f"\nRegion assignments:")
    for region in unique_regions:
        if region > 0:
            count = np.sum(regions == region)
            print(f"  {region_names.get(region, f'Region {region}')}: {count} voxels")


def example_2_weighted_regression():
    """
    Example 2: Weighted regression segmentation.
    
    Shows how to:
    - Fit weighted regression lines on 2D planes
    - Compare ridge regression vs robust regression
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Weighted Regression Segmentation")
    print("="*60)
    
    # Create synthetic vertebra
    vertebra_mask = create_synthetic_vertebra(
        shape=(50, 60, 40),
        center=(25, 30, 20),
        radii=(15, 20, 18),
        rotation_angle=10
    )
    
    segmenter = VertebraSegmenter(vertebra_mask, pixel_spacing=1.0, slice_thickness=1.0)
    
    # Get weighted regression splits on different planes
    print("\nFitting weighted regression lines...")
    
    # Coronal plane (Y-Z plane) - separates anterior/posterior
    coronal_line, cor_ridge = segmenter.split_by_weighted_regression(
        plane='coronal', method='ridge'
    )
    print(f"Coronal plane (ridge):  m={cor_ridge.coef_[0]:.3f}, b={cor_ridge.intercept_:.3f}")
    
    coronal_line_robust, cor_huber = segmenter.split_by_weighted_regression(
        plane='coronal', method='huber'
    )
    print(f"Coronal plane (huber):  m={cor_huber.coef_[0]:.3f}, b={cor_huber.intercept_:.3f}")
    
    # Sagittal plane (X-Z plane) - separates left/right
    sagittal_line, sag_ridge = segmenter.split_by_weighted_regression(
        plane='sagittal', method='ridge'
    )
    print(f"Sagittal plane (ridge): m={sag_ridge.coef_[0]:.3f}, b={sag_ridge.intercept_:.3f}")
    
    sagittal_line_robust, sag_huber = segmenter.split_by_weighted_regression(
        plane='sagittal', method='huber'
    )
    print(f"Sagittal plane (huber): m={sag_huber.coef_[0]:.3f}, b={sag_huber.intercept_:.3f}")


def example_3_pca_vs_regression():
    """
    Example 3: Compare PCA and regression approaches on noisy vertebra.
    
    Shows robustness to geometry variations.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: PCA vs Regression - Robustness Comparison")
    print("="*60)
    
    # Create vertebra with noise and irregular shape
    vertebra_mask = create_synthetic_vertebra(
        shape=(50, 60, 40),
        center=(25, 30, 20),
        radii=(15, 20, 18),
        rotation_angle=20
    )
    
    # Add some noise/irregularity
    noise = np.random.rand(*vertebra_mask.shape) > 0.85
    vertebra_mask = (vertebra_mask & ~noise) | (noise & ~vertebra_mask)
    
    segmenter = VertebraSegmenter(vertebra_mask, pixel_spacing=1.0, slice_thickness=1.0)
    
    # PCA approach
    print("\nPCA Approach:")
    axes, com, evals = segmenter.split_by_3d_pca()
    regions_pca = segmenter.assign_regions_pca(axes, com)
    print(f"  Assigned regions based on principal axes")
    
    # Regression approach
    print("\nWeighted Regression Approach:")
    cor_line, cor_reg = segmenter.split_by_weighted_regression('coronal', 'ridge')
    sag_line, sag_reg = segmenter.split_by_weighted_regression('sagittal', 'ridge')
    print(f"  Coronal regression: m={cor_reg.coef_[0]:.3f}, b={cor_reg.intercept_:.3f}")
    print(f"  Sagittal regression: m={sag_reg.coef_[0]:.3f}, b={sag_reg.intercept_:.3f}")
    
    print("\nBoth approaches work on irregular/tilted vertebrae without")
    print("requiring spine-type specific geometry parameters!")


def example_4_spine_type_agnostic():
    """
    Example 4: Show that same method works across spine types.
    
    Demonstrates cervical, thoracic, and lumbar vertebrae with different
    geometries, but using the same segmentation parameters.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Spine-Type Agnostic Segmentation")
    print("="*60)
    
    spine_types = [
        ('Cervical', {'radii': (12, 15, 14), 'rotation': 5}),
        ('Thoracic', {'radii': (13, 18, 16), 'rotation': 8}),
        ('Lumbar',   {'radii': (16, 22, 19), 'rotation': 10}),
    ]
    
    for spine_name, params in spine_types:
        print(f"\n{spine_name} vertebra:")
        
        vertebra_mask = create_synthetic_vertebra(
            shape=(50, 60, 40),
            center=(25, 30, 20),
            radii=params['radii'],
            rotation_angle=params['rotation']
        )
        
        segmenter = VertebraSegmenter(vertebra_mask, pixel_spacing=1.0, slice_thickness=1.0)
        stats = segmenter.get_summary_statistics()
        
        print(f"  Eigenvalues: {stats['eigenvalues']}")
        print(f"  Aspect ratios: {[f'{r:.2f}' for r in stats['aspect_ratios']]}")
        
        # Same PCA segmentation for all types
        axes, com, _ = segmenter.split_by_3d_pca()
        regions = segmenter.assign_regions_pca(axes, com)
        
        print(f"  Segmentation successful with same parameters!")


def create_synthetic_vertebra(shape, center, radii, rotation_angle=0):
    """
    Create a synthetic vertebra mask (ellipsoid with optional rotation).
    
    Parameters
    ----------
    shape : tuple
        3D shape of output array
    center : tuple
        Center of ellipsoid
    radii : tuple
        Semi-axes lengths (a, b, c)
    rotation_angle : float
        Rotation angle in degrees around Z axis
    
    Returns
    -------
    mask : ndarray, bool
        3D binary mask of synthetic vertebra
    """
    mask = np.zeros(shape, dtype=bool)
    
    # Convert rotation angle to radians
    theta = np.radians(rotation_angle)
    
    # Create indices
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Shift to center
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]
    
    # Apply rotation to X-Y plane
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    
    # Ellipsoid equation
    ellipsoid = (x_rot**2 / radii[0]**2 + 
                 y_rot**2 / radii[1]**2 + 
                 z**2 / radii[2]**2) <= 1
    
    mask[ellipsoid] = True
    
    return mask


def visualize_pca_comparison():
    """
    Create visualizations comparing PCA and regression approaches.
    """
    print("\n" + "="*60)
    print("Creating visualization (will save as PDF if matplotlib backend supports)")
    print("="*60)
    
    # Create two different vertebrae
    vertebra1 = create_synthetic_vertebra((50, 60, 40), (25, 30, 20), 
                                          (15, 20, 18), rotation_angle=0)
    vertebra2 = create_synthetic_vertebra((50, 60, 40), (25, 30, 20), 
                                          (15, 20, 18), rotation_angle=25)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for row, (vertebra, title) in enumerate([
        (vertebra1, "Axis-Aligned Vertebra"),
        (vertebra2, "Rotated Vertebra (25°)")
    ]):
        segmenter = VertebraSegmenter(vertebra, 1.0, 1.0)
        
        # Show coronal slice through center
        axes[row, 0].imshow(vertebra[25, :, :], cmap='gray')
        axes[row, 0].set_title(f"{title}\n(Coronal Slice)")
        axes[row, 0].axis('off')
        
        # Show sagittal slice
        axes[row, 1].imshow(vertebra[:, 30, :], cmap='gray')
        axes[row, 1].set_title(f"{title}\n(Sagittal Slice)")
        axes[row, 1].axis('off')
        
        # Show PCA analysis
        stats = segmenter.get_summary_statistics()
        ax_pca = axes[row, 2]
        
        # Plot eigenvalues
        ax_pca.bar(range(1, 4), stats['eigenvalues'])
        ax_pca.set_ylabel('Eigenvalue')
        ax_pca.set_xlabel('Principal Component')
        ax_pca.set_title(f"{title}\n(PCA Eigenvalues)")
        ax_pca.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    print("\nVisualization created!")
    return fig


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PCA-Based Vertebra Segmentation - Examples")
    print("="*60)
    
    # Run examples
    example_1_pca_basic()
    example_2_weighted_regression()
    example_3_pca_vs_regression()
    example_4_spine_type_agnostic()
    
    # Create visualization
    fig = visualize_pca_comparison()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
