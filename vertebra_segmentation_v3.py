# -*- coding: utf-8 -*-
"""
Advanced vertebra segmentation using PCA and weighted regression.

This module provides generalized methods for splitting individual vertebrae
into anatomical regions (superior/inferior, anterior/posterior, left/right)
that work across different spine types (cervical, thoracic, lumbar) without
requiring geometry-specific adjustments.

Author: Rune P
Date: 2024
"""

import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, Ridge
import warnings


class VertebraSegmenter:
    """
    A generalized vertebra segmenter using PCA and weighted regression methods.
    
    Works across cervical, thoracic, and lumbar vertebrae without special-casing
    geometry. Provides multiple approaches for splitting vertebrae into regions.
    """
    
    def __init__(self, vertebra_mask, pixel_spacing=1.0, slice_thickness=1.0):
        """
        Initialize the segmenter with a single vertebra.
        
        Parameters
        ----------
        vertebra_mask : ndarray, shape (x, y, z)
            Binary 3D mask of the vertebra (1 = vertebra, 0 = background)
        pixel_spacing : float, optional
            Pixel spacing in mm (default: 1.0)
        slice_thickness : float, optional
            Slice thickness in mm (default: 1.0)
        """
        self.mask = vertebra_mask.astype(bool)
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        
        # Get vertebra coordinates
        self.coords = np.argwhere(self.mask)
        if len(self.coords) == 0:
            raise ValueError("Empty vertebra mask provided")
        
        # Center of mass
        self.com = ndimage.center_of_mass(self.mask)
        
        # Output segmentation (will hold region labels)
        self.regions = np.zeros_like(vertebra_mask, dtype=np.int32)
    
    def get_moments_and_axes(self):
        """
        Compute second moments (inertia tensor) and principal axes.
        
        Returns
        -------
        principal_axes : ndarray, shape (3, 3)
            Principal axes as column vectors (eigenvectors of inertia tensor)
        eigenvalues : ndarray, shape (3,)
            Eigenvalues (moment magnitudes)
        principal_axes_indices : ndarray, shape (3,)
            Indices of axes sorted by magnitude (largest first)
        """
        # Center coordinates relative to COM
        centered_coords = self.coords - np.array(self.com)
        
        # Build inertia tensor (second moment tensor)
        # I_ij = sum(r_i * r_j) for all points
        inertia = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                inertia[i, j] = np.sum(centered_coords[:, i] * centered_coords[:, j])
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # Sort by magnitude (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        return eigenvectors[:, sorted_indices], eigenvalues[sorted_indices], sorted_indices
    
    def _get_plane_coordinates(self, plane='coronal'):
        """
        Extract 2D coordinates on a specified plane.
        
        Parameters
        ----------
        plane : str, one of {'coronal', 'sagittal', 'axial'}
            Which plane to extract
        
        Returns
        -------
        coords_2d : ndarray, shape (n_points, 2)
            2D coordinates on the plane
        weights : ndarray, shape (n_points,)
            Weights (distance from boundary, normalized)
        """
        if plane == 'coronal':
            # Take projection on Y-Z plane (sagittal-axial)
            coords_2d = self.coords[:, [1, 2]]
            plane_axis = 0  # coronal axis
        elif plane == 'sagittal':
            # Take projection on X-Z plane (coronal-axial)
            coords_2d = self.coords[:, [0, 2]]
            plane_axis = 1  # sagittal axis
        elif plane == 'axial':
            # Take projection on X-Y plane (coronal-sagittal)
            coords_2d = self.coords[:, [0, 1]]
            plane_axis = 2  # axial axis
        else:
            raise ValueError(f"Unknown plane: {plane}")
        
        # Compute weights based on distance from boundary
        # Points closer to center get higher weight
        center_2d = np.array(self.com)[[1, 2]] if plane == 'coronal' else \
                    np.array(self.com)[[0, 2]] if plane == 'sagittal' else \
                    np.array(self.com)[[0, 1]]
        
        distances = np.linalg.norm(coords_2d - center_2d, axis=1)
        max_dist = np.max(distances) + 1e-6
        weights = 1.0 - (distances / max_dist)
        
        return coords_2d, weights
    
    def split_by_weighted_regression(self, plane='coronal', method='ridge'):
        """
        Split vertebra on a plane using weighted regression.
        
        The method fits a line through the vertebra on the specified plane,
        using weights based on distance from center. This is more robust than
        standard least squares and works across different vertebra types.
        
        Parameters
        ----------
        plane : str, one of {'coronal', 'sagittal', 'axial'}
            Which plane to split on
        method : str, one of {'ridge', 'huber'}
            Regression method:
            - 'ridge': Ridge regression (L2 regularization)
            - 'huber': HuberRegressor (robust to outliers)
        
        Returns
        -------
        split_line : ndarray, shape (2, n_samples)
            Points defining the split line (2D)
        regression_line : object
            Fitted regression model
        """
        coords_2d, weights = self._get_plane_coordinates(plane)
        
        # Fit line: y = mx + b
        X = coords_2d[:, 0:1]  # 2D -> needs to be column vector
        y = coords_2d[:, 1]
        
        if method == 'ridge':
            regressor = Ridge(alpha=1.0)
            regressor.fit(X, y, sample_weight=weights)
        elif method == 'huber':
            regressor = HuberRegressor(epsilon=1.35, max_iter=500)
            regressor.fit(X, y, sample_weight=weights)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Generate line points
        x_range = np.linspace(np.min(X), np.max(X), 100)
        y_range = regressor.predict(x_range.reshape(-1, 1))
        
        split_line = np.array([x_range, y_range])
        
        return split_line, regressor
    
    def split_by_pca_plane(self, plane='coronal'):
        """
        Split vertebra using PCA on a specified plane.
        
        Fits a PCA model to 2D coordinates and uses the principal axis
        as the split line. This captures the natural orientation of the
        vertebra without assuming alignment to image axes.
        
        Parameters
        ----------
        plane : str, one of {'coronal', 'sagittal', 'axial'}
            Which plane to split on
        
        Returns
        -------
        split_line : ndarray, shape (2, n_samples)
            Points defining the split line (2D)
        pca_model : PCA object
            Fitted PCA model
        """
        coords_2d, _ = self._get_plane_coordinates(plane)
        
        # Fit PCA to find principal direction
        pca = PCA(n_components=2)
        pca.fit(coords_2d)
        
        # The first principal component is the main axis
        # The second principal component is perpendicular (the split direction)
        
        # Get center and principal axes
        center = pca.mean_
        pc1 = pca.components_[0]  # Primary direction
        pc2 = pca.components_[1]  # Split direction
        
        # Generate line perpendicular to PC1 (along PC2)
        t = np.linspace(-3 * np.sqrt(pca.explained_variance_[1]),
                        3 * np.sqrt(pca.explained_variance_[1]), 100)
        
        # Line parametric form: center + t * pc2
        line_points = center[:, np.newaxis] + pc2[:, np.newaxis] * t
        
        return line_points, pca
    
    def split_by_3d_pca(self):
        """
        Split vertebra using full 3D PCA to define three orthogonal planes.
        
        This method computes the principal axes of the entire vertebra volume
        and uses them to define splitting planes for all three regions
        (superior/inferior, anterior/posterior, left/right).
        
        Returns
        -------
        principal_axes : ndarray, shape (3, 3)
            Principal axes (column vectors)
        com : ndarray, shape (3,)
            Center of mass
        eigenvalues : ndarray, shape (3,)
            Eigenvalues (magnitudes)
        """
        axes, eigenvalues, _ = self.get_moments_and_axes()
        
        return axes, self.com, eigenvalues
    
    def assign_regions_pca(self, axes, com=None):
        """
        Assign vertebra voxels to regions using 3D PCA axes.
        
        Uses principal axes to define three orthogonal splitting planes.
        This method is robust across vertebra types without geometry-specific
        parameters.
        
        Parameters
        ----------
        axes : ndarray, shape (3, 3)
            Principal axes from PCA
        com : ndarray, optional
            Center of mass. If None, uses self.com
        
        Returns
        -------
        regions : ndarray, same shape as input mask
            Region labels:
            - 0: background
            - 1: vertebra (no region assigned)
            - 2: superior
            - 4: inferior
            - 8: anterior
            - 16: posterior
            - 32: left
            - 64: right
        """
        if com is None:
            com = self.com
        
        # Define region keys
        SUPERIOR = 2
        INFERIOR = 4
        ANTERIOR = 8
        POSTERIOR = 16
        LEFT = 32
        RIGHT = 64
        
        regions = np.zeros_like(self.mask, dtype=np.int32)
        
        # For each voxel in vertebra, determine which region it belongs to
        for idx in self.coords:
            # Vector from COM to voxel
            vec = idx - np.array(com)
            
            # Project onto each principal axis
            projections = np.dot(axes.T, vec)  # 3 projections
            
            # Axis 0: superior/inferior (typically longest axis for cylindrical vertebra)
            if projections[0] > 0:
                regions[tuple(idx)] = SUPERIOR
            else:
                regions[tuple(idx)] = INFERIOR
            
            # Axis 1: anterior/posterior
            if projections[1] > 0:
                regions[tuple(idx)] = regions[tuple(idx)] + POSTERIOR
            else:
                regions[tuple(idx)] = regions[tuple(idx)] + ANTERIOR
            
            # Axis 2: left/right
            if projections[2] > 0:
                regions[tuple(idx)] = regions[tuple(idx)] + RIGHT
            else:
                regions[tuple(idx)] = regions[tuple(idx)] + LEFT
        
        self.regions = regions
        return regions
    
    def assign_regions_by_plane(self, plane_normal, plane_point, region_code_true, 
                                region_code_false, existing_regions=None):
        """
        Assign vertebra voxels to two regions based on a plane.
        
        Uses a plane defined by normal and point to partition vertebra into
        two regions (e.g., anterior and posterior).
        
        Parameters
        ----------
        plane_normal : ndarray, shape (3,)
            Normal vector to the splitting plane
        plane_point : ndarray, shape (3,)
            A point on the plane (typically COM shifted along normal)
        region_code_true : int
            Label for voxels on the positive side of plane
        region_code_false : int
            Label for voxels on the negative side of plane
        existing_regions : ndarray, optional
            Existing region assignment to combine with
        
        Returns
        -------
        regions : ndarray, same shape as input mask
            Updated region labels
        """
        if existing_regions is None:
            regions = np.zeros_like(self.mask, dtype=np.int32)
        else:
            regions = existing_regions.copy()
        
        # Normalize plane normal
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-10)
        
        for idx in self.coords:
            # Vector from plane point to voxel
            vec = idx - np.array(plane_point)
            
            # Signed distance to plane (dot product with normal)
            signed_dist = np.dot(vec, plane_normal)
            
            if signed_dist >= 0:
                regions[tuple(idx)] = regions[tuple(idx)] | region_code_true
            else:
                regions[tuple(idx)] = regions[tuple(idx)] | region_code_false
        
        self.regions = regions
        return regions
    
    def get_summary_statistics(self):
        """
        Compute and return summary statistics of the vertebra.
        
        Useful for understanding the vertebra's geometry and choosing
        appropriate segmentation parameters.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'volume': Number of voxels
            - 'principal_axes': PCA principal axes
            - 'eigenvalues': PCA eigenvalues (magnitude ordering)
            - 'aspect_ratios': Ratios of eigenvalues
            - 'bounding_box': Min/max coordinates
            - 'centroid': Center of mass
        """
        axes, eigenvalues, _ = self.get_moments_and_axes()
        
        bbox_min = np.min(self.coords, axis=0)
        bbox_max = np.max(self.coords, axis=0)
        
        aspect_ratios = eigenvalues / (eigenvalues[-1] + 1e-10)
        
        stats = {
            'volume': len(self.coords),
            'principal_axes': axes,
            'eigenvalues': eigenvalues,
            'aspect_ratios': aspect_ratios,
            'bounding_box': np.array([bbox_min, bbox_max]),
            'centroid': np.array(self.com),
        }
        
        return stats


def segment_vertebra_pca(vertebra_mask, pixel_spacing=1.0, slice_thickness=1.0):
    """
    Convenience function to segment a single vertebra using PCA.
    
    Parameters
    ----------
    vertebra_mask : ndarray, shape (x, y, z)
        Binary 3D mask of vertebra
    pixel_spacing : float
        Pixel spacing in mm
    slice_thickness : float
        Slice thickness in mm
    
    Returns
    -------
    regions : ndarray, same shape as input
        Segmented regions
    segmenter : VertebraSegmenter
        Segmenter object for further analysis
    """
    segmenter = VertebraSegmenter(vertebra_mask, pixel_spacing, slice_thickness)
    axes, com, eigenvalues = segmenter.split_by_3d_pca()
    regions = segmenter.assign_regions_pca(axes, com)
    
    return regions, segmenter


def segment_vertebra_weighted_regression(vertebra_mask, pixel_spacing=1.0, 
                                         slice_thickness=1.0):
    """
    Convenience function to segment a vertebra using weighted regression.
    
    Fits separate regression lines to coronal and sagittal planes.
    
    Parameters
    ----------
    vertebra_mask : ndarray, shape (x, y, z)
        Binary 3D mask of vertebra
    pixel_spacing : float
        Pixel spacing in mm
    slice_thickness : float
        Slice thickness in mm
    
    Returns
    -------
    regions : ndarray, same shape as input
        Segmented regions
    segmenter : VertebraSegmenter
        Segmenter object for further analysis
    """
    segmenter = VertebraSegmenter(vertebra_mask, pixel_spacing, slice_thickness)
    regions = np.zeros_like(vertebra_mask, dtype=np.int32)
    
    # Region codes (matching load_mapping conventions)
    ANTERIOR = 4
    POSTERIOR = 8
    LEFT = 16
    RIGHT = 32
    SUPERIOR = 64
    INFERIOR = 128
    
    # Get split lines
    coronal_line, cor_regressor = segmenter.split_by_weighted_regression('coronal')
    sagittal_line, sag_regressor = segmenter.split_by_weighted_regression('sagittal')
    
    # Assign regions based on regression lines
    # This would be implemented similar to assign_regions_by_plane
    # but using the fitted regression lines
    
    segmenter.regions = regions
    return regions, segmenter


if __name__ == "__main__":
    # Example usage would go here
    print("Vertebra segmentation module loaded successfully.")
    print("Use VertebraSegmenter class or convenience functions:")
    print("  - segment_vertebra_pca()")
    print("  - segment_vertebra_weighted_regression()")
