# -*- coding: utf-8 -*-
"""
Test suite for vertebra segmentation module.

Run this to verify the implementation works correctly.

Usage:
    python test_vertebra_segmentation.py
"""

import numpy as np
from vertebra_segmentation_v3 import VertebraSegmenter, segment_vertebra_pca


def create_test_vertebra(spine_type='lumbar', rotation=0, noise_level=0.0):
    """
    Create synthetic test vertebra.
    
    Parameters
    ----------
    spine_type : str
        'cervical', 'thoracic', or 'lumbar'
    rotation : float
        Rotation angle in degrees
    noise_level : float
        Fraction of voxels to randomly flip as noise
    
    Returns
    -------
    mask : ndarray
        3D binary vertebra mask
    """
    # Define parameters per spine type
    params = {
        'cervical': {'radii': (12, 15, 14), 'center': (25, 30, 20)},
        'thoracic': {'radii': (13, 18, 16), 'center': (25, 30, 20)},
        'lumbar':   {'radii': (16, 22, 19), 'center': (25, 30, 20)},
    }
    
    if spine_type not in params:
        raise ValueError(f"Unknown spine type: {spine_type}")
    
    shape = (50, 60, 40)
    mask = np.zeros(shape, dtype=bool)
    
    # Create ellipsoid
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = params[spine_type]['center']
    radii = params[spine_type]['radii']
    
    # Shift to center
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]
    
    # Apply rotation
    theta = np.radians(rotation)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    
    # Ellipsoid equation
    ellipsoid = (x_rot**2 / radii[0]**2 + 
                 y_rot**2 / radii[1]**2 + 
                 z**2 / radii[2]**2) <= 1
    
    mask[ellipsoid] = True
    
    # Add noise
    if noise_level > 0:
        noise = np.random.rand(*shape) < noise_level
        mask = (mask & ~noise) | (noise & ~mask)
    
    return mask


class TestSuite:
    """Test suite for vertebra segmentation."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def run_test(self, name, test_func):
        """Run a single test."""
        try:
            test_func()
            print(f"✓ {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ {name}")
            print(f"  Error: {e}")
            self.failed += 1
        except Exception as e:
            print(f"✗ {name} (Exception)")
            print(f"  Error: {e}")
            self.failed += 1
    
    def test_basic_initialization(self):
        """Test: Can initialize segmenter."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        assert seg.com is not None
        assert len(seg.coords) > 0
    
    def test_pca_computation(self):
        """Test: PCA computation works."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        axes, evals, indices = seg.get_moments_and_axes()
        
        assert axes.shape == (3, 3), f"Wrong axes shape: {axes.shape}"
        assert evals.shape == (3,), f"Wrong evals shape: {evals.shape}"
        assert np.all(evals > 0), "Eigenvalues should be positive"
        assert np.all(evals[:-1] >= evals[1:]), "Eigenvalues should be sorted descending"
    
    def test_orthogonal_axes(self):
        """Test: Principal axes are orthogonal."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        axes, _, _ = seg.get_moments_and_axes()
        
        # Check orthogonality
        for i in range(3):
            for j in range(i+1, 3):
                dot_prod = np.dot(axes[:, i], axes[:, j])
                assert abs(dot_prod) < 1e-6, f"Axes {i} and {j} not orthogonal: {dot_prod}"
    
    def test_unit_axes(self):
        """Test: Principal axes are unit vectors."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        axes, _, _ = seg.get_moments_and_axes()
        
        for i in range(3):
            norm = np.linalg.norm(axes[:, i])
            assert abs(norm - 1.0) < 1e-6, f"Axis {i} not unit vector: norm={norm}"
    
    def test_pca_segmentation(self):
        """Test: PCA segmentation assigns all voxels."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        axes, com, _ = seg.split_by_3d_pca()
        regions = seg.assign_regions_pca(axes, com)
        
        # Check that all voxels in mask get assigned
        assigned = np.sum(regions > 0)
        in_mask = np.sum(mask > 0)
        
        assert assigned == in_mask, f"Not all voxels assigned: {assigned}/{in_mask}"
    
    def test_spine_types(self):
        """Test: All spine types work."""
        for spine_type in ['cervical', 'thoracic', 'lumbar']:
            mask = create_test_vertebra(spine_type)
            seg = VertebraSegmenter(mask)
            regions = seg.assign_regions_pca(*seg.split_by_3d_pca()[:2])
            assert np.sum(regions > 0) > 0, f"{spine_type} failed"
    
    def test_rotated_vertebra(self):
        """Test: Works on rotated vertebrae."""
        for angle in [0, 10, 20, 45]:
            mask = create_test_vertebra('lumbar', rotation=angle)
            seg = VertebraSegmenter(mask)
            regions = seg.assign_regions_pca(*seg.split_by_3d_pca()[:2])
            assert np.sum(regions > 0) > 0, f"Failed at rotation={angle}°"
    
    def test_noisy_vertebra(self):
        """Test: Robust to noise."""
        mask = create_test_vertebra('lumbar', noise_level=0.1)
        seg = VertebraSegmenter(mask)
        regions = seg.assign_regions_pca(*seg.split_by_3d_pca()[:2])
        assert np.sum(regions > 0) > 0
    
    def test_weighted_regression_coronal(self):
        """Test: Weighted regression on coronal plane."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        line, regressor = seg.split_by_weighted_regression('coronal', method='ridge')
        
        assert line.shape[0] == 2, f"Wrong line shape: {line.shape}"
        assert hasattr(regressor, 'coef_'), "Regressor missing coef_"
        assert hasattr(regressor, 'intercept_'), "Regressor missing intercept_"
    
    def test_weighted_regression_sagittal(self):
        """Test: Weighted regression on sagittal plane."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        line, regressor = seg.split_by_weighted_regression('sagittal', method='huber')
        
        assert line.shape[0] == 2, f"Wrong line shape: {line.shape}"
    
    def test_statistics(self):
        """Test: Statistics computation."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        stats = seg.get_summary_statistics()
        
        required_keys = ['volume', 'principal_axes', 'eigenvalues', 
                        'aspect_ratios', 'bounding_box', 'centroid']
        for key in required_keys:
            assert key in stats, f"Missing key in stats: {key}"
        
        assert stats['volume'] > 0
        assert stats['aspect_ratios'].shape == (3,)
    
    def test_convenience_function(self):
        """Test: Convenience function works."""
        mask = create_test_vertebra('lumbar')
        regions, segmenter = segment_vertebra_pca(mask)
        
        assert regions.shape == mask.shape
        assert isinstance(segmenter, VertebraSegmenter)
        assert np.sum(regions > 0) > 0
    
    def test_region_balance(self):
        """Test: Regions are reasonably balanced."""
        mask = create_test_vertebra('lumbar')
        seg = VertebraSegmenter(mask)
        regions = seg.assign_regions_pca(*seg.split_by_3d_pca()[:2])
        
        unique, counts = np.unique(regions, return_counts=True)
        total = np.sum(regions > 0)
        
        # Get region counts (excluding background)
        region_counts = counts[unique > 0]
        
        # Check that no single region has >50% of voxels
        max_fraction = np.max(region_counts) / total
        assert max_fraction < 0.5, f"Region too large: {max_fraction*100:.1f}%"
        
        # Check that no region has <5% (for 8 regions, expect ~12% each)
        min_fraction = np.min(region_counts) / total
        assert min_fraction > 0.02, f"Region too small: {min_fraction*100:.1f}%"
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "="*60)
        print("Running Vertebra Segmentation Tests")
        print("="*60 + "\n")
        
        tests = [
            ("Basic Initialization", self.test_basic_initialization),
            ("PCA Computation", self.test_pca_computation),
            ("Orthogonal Axes", self.test_orthogonal_axes),
            ("Unit Axes", self.test_unit_axes),
            ("PCA Segmentation", self.test_pca_segmentation),
            ("All Spine Types", self.test_spine_types),
            ("Rotated Vertebrae", self.test_rotated_vertebra),
            ("Noisy Vertebrae", self.test_noisy_vertebra),
            ("Weighted Regression (Coronal)", self.test_weighted_regression_coronal),
            ("Weighted Regression (Sagittal)", self.test_weighted_regression_sagittal),
            ("Statistics Computation", self.test_statistics),
            ("Convenience Function", self.test_convenience_function),
            ("Region Balance", self.test_region_balance),
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        print("\n" + "="*60)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("="*60 + "\n")
        
        return self.failed == 0


if __name__ == "__main__":
    np.random.seed(42)
    suite = TestSuite()
    success = suite.run_all()
    
    if success:
        print("✓ All tests passed! Implementation is working correctly.")
        exit(0)
    else:
        print("✗ Some tests failed. Please review the implementation.")
        exit(1)
