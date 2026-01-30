"""
Stage 4: Scale-Space Consistency Analysis

Analyzes anomaly persistence across scale space.

Rationale: True features persist across multiple scales (image pyramid
levels). Noise and artifacts tend to disappear at coarser scales.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import List, Dict


class ScaleSpaceConsistencyAnalyzer:
    """
    Analyzes anomaly persistence across scale space.
    
    Rationale: True features persist across multiple scales (image pyramid
    levels). Noise and artifacts tend to disappear at coarser scales.
    """
    
    def __init__(self, num_scales: int = 4, sigma_base: float = 1.0):
        """
        Args:
            num_scales: Number of pyramid levels
            sigma_base: Base Gaussian sigma for pyramid
        """
        self.num_scales = num_scales
        self.sigma_base = sigma_base
    
    def build_gaussian_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Build Gaussian pyramid for scale-space analysis."""
        pyramid = [image]
        current = image.copy()
        
        for i in range(1, self.num_scales):
            # Apply Gaussian blur and downsample
            sigma = self.sigma_base * (2 ** i)
            blurred = gaussian_filter(current, sigma=sigma)
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)
            current = downsampled
        
        return pyramid
    
    def detect_at_scale(self, image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Detect anomalies at a specific scale using edge detection.
        """
        # Compute gradient magnitude
        dy, dx = np.gradient(image)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Threshold
        detection = (gradient_mag > threshold).astype(np.float32)
        
        return detection
    
    def analyze_scale_consistency(self, inconsistency_map: np.ndarray) -> Dict:
        """Analyze how anomalies persist across scales."""
        # Build pyramid of inconsistency map
        pyramid = self.build_gaussian_pyramid(inconsistency_map)
        
        # Detect at each scale
        detections = []
        for level, image in enumerate(pyramid):
            detection = self.detect_at_scale(image, threshold=0.3)
            detections.append(detection)
        
        # Compute scale persistence
        # Upscale all detections to original size and accumulate
        original_shape = inconsistency_map.shape
        persistence_map = np.zeros(original_shape, dtype=np.float32)
        
        for level, detection in enumerate(detections):
            # Upscale to original size
            upscaled = cv2.resize(detection, 
                                (original_shape[1], original_shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            
            # Weight by scale (coarser scales more important)
            weight = 1.0 / (level + 1)
            persistence_map += upscaled * weight
        
        # Normalize
        total_weight = sum(1.0 / (i + 1) for i in range(len(detections)))
        persistence_map /= total_weight
        
        # Final consistency score
        consistency_score = persistence_map * inconsistency_map
        
        return {
            'pyramid': pyramid,
            'detections': detections,
            'persistence_map': persistence_map,
            'consistency_score': consistency_score
        }
