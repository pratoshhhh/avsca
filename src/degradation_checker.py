"""
Stage 6: Physical Degradation Consistency Check

Verifies that anomaly regions degrade consistently with background.

Rationale: Anomalies should degrade (blur, noise) similarly to the rest
of the image. Foreign objects or digital artifacts may exhibit
inconsistent degradation patterns.
"""

import numpy as np
import cv2
from typing import Dict
from .image_degradation import ImageDegradationEstimator


class PhysicalDegradationChecker:
    """
    Verifies that anomaly regions degrade consistently with background.
    
    Rationale: Anomalies should degrade (blur, noise) similarly to the rest
    of the image. Foreign objects or digital artifacts may exhibit
    inconsistent degradation patterns.
    """
    
    def __init__(self, tolerance_sigma: float = 2.0):
        """
        Args:
            tolerance_sigma: Threshold for degradation anomaly detection
        """
        self.tolerance_sigma = tolerance_sigma
        self.degradation_estimator = ImageDegradationEstimator()
    
    def compute_local_degradation(self, 
                                 frame: np.ndarray,
                                 mask: np.ndarray) -> Dict:
        """Compute degradation metrics for masked regions."""
        # Extract region pixels
        if mask.sum() == 0:
            return {
                'blur': 0.0,
                'noise': 0.0,
                'contrast': 0.0
            }
        
        # Create masked frame
        masked_frame = frame * mask
        
        # Compute local metrics
        region_pixels = frame[mask > 0]
        
        # Local noise estimate
        local_noise = np.std(region_pixels)
        
        # Local contrast
        local_contrast = region_pixels.std()
        
        # Local blur (using edge density)
        edges = cv2.Canny((frame * 255).astype(np.uint8), 50, 150)
        edge_density = (edges[mask > 0] > 0).sum() / mask.sum()
        local_blur = 1.0 - edge_density  # High blur = low edge density
        
        return {
            'blur': local_blur,
            'noise': local_noise,
            'contrast': local_contrast
        }
    
    def check_degradation_consistency(self,
                                     frame: np.ndarray,
                                     candidate_mask: np.ndarray,
                                     global_metrics: Dict) -> Dict:
        """
        Check if degradation in candidate regions is consistent with global frame.
        """
        # Compute local degradation
        local_metrics = self.compute_local_degradation(frame, candidate_mask)
        
        # Compare with global metrics
        blur_diff = abs(local_metrics['blur'] - global_metrics['blur']) / (global_metrics['blur'] + 1e-6)
        noise_diff = abs(local_metrics['noise'] - global_metrics['noise']) / (global_metrics['noise'] + 1e-6)
        contrast_diff = abs(local_metrics['contrast'] - global_metrics['contrast']) / (global_metrics['contrast'] + 1e-6)
        
        # Aggregate inconsistency score
        inconsistency = (blur_diff + noise_diff + contrast_diff) / 3.0
        
        # Consistency weight (1.0 = fully consistent, 0.0 = highly inconsistent)
        consistency_weight = np.exp(-inconsistency * self.tolerance_sigma)
        
        return {
            'local_metrics': local_metrics,
            'inconsistency': inconsistency,
            'consistency_weight': consistency_weight,
            'differences': {
                'blur': blur_diff,
                'noise': noise_diff,
                'contrast': contrast_diff
            }
        }
