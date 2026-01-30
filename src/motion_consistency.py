"""
Stage 3: Motion Self-Consistency Check

Identifies motion anomalies based on residual motion analysis.

Rationale: Regions with residual motion significantly different from zero
violate the assumption of static background.
"""

import numpy as np
import cv2
from typing import Dict


class MotionConsistencyChecker:
    """
    Identifies motion anomalies based on residual motion analysis.
    
    Rationale: Regions with residual motion significantly different from zero
    violate the assumption of static background.
    """
    
    def __init__(self, threshold_sigma: float = 3.0, min_area: int = 100):
        """
        Args:
            threshold_sigma: Number of standard deviations for anomaly threshold
            min_area: Minimum area (pixels) for valid anomaly regions
        """
        self.threshold_sigma = threshold_sigma
        self.min_area = min_area
    
    def compute_motion_inconsistency(self, residual_magnitude: np.ndarray) -> Dict:
        """Compute motion inconsistency map from residual motion."""
        # Robust statistics (using median/MAD to handle outliers)
        median_residual = np.median(residual_magnitude)
        mad = np.median(np.abs(residual_magnitude - median_residual))
        sigma_estimate = 1.4826 * mad  # Convert MAD to std estimate
        
        # Adaptive threshold
        threshold = median_residual + self.threshold_sigma * sigma_estimate
        
        # Binary anomaly mask
        anomaly_mask = (residual_magnitude > threshold).astype(np.uint8)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anomaly_mask, connectivity=8
        )
        
        filtered_mask = np.zeros_like(anomaly_mask)
        valid_regions = []
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                filtered_mask[labels == i] = 1
                valid_regions.append({
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': (stats[i, cv2.CC_STAT_LEFT],
                            stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT])
                })
        
        # Continuous inconsistency score (normalized residual)
        inconsistency_score = np.clip(
            (residual_magnitude - median_residual) / (sigma_estimate + 1e-6),
            0, 10
        ) / 10.0  # Normalize to [0, 1]
        
        return {
            'inconsistency_map': inconsistency_score,
            'binary_mask': filtered_mask,
            'threshold': threshold,
            'num_regions': len(valid_regions),
            'regions': valid_regions,
            'statistics': {
                'median': median_residual,
                'sigma': sigma_estimate,
                'max_residual': residual_magnitude.max()
            }
        }
