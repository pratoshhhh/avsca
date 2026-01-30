"""
Stage 7: Fusion & Final Anomaly Map

Fuses multiple consistency scores into final anomaly map.
"""

import numpy as np
import cv2
from matplotlib import cm
from typing import Dict


class AnomalyFusionEngine:
    """
    Fuses multiple consistency scores into final anomaly map.
    """
    
    def __init__(self, 
                 motion_weight: float = 0.3,
                 scale_weight: float = 0.25,
                 temporal_weight: float = 0.3,
                 degradation_weight: float = 0.15):
        """
        Args:
            motion_weight: Weight for motion consistency
            scale_weight: Weight for scale consistency
            temporal_weight: Weight for temporal consistency
            degradation_weight: Weight for degradation consistency
        """
        total = motion_weight + scale_weight + temporal_weight + degradation_weight
        self.weights = {
            'motion': motion_weight / total,
            'scale': scale_weight / total,
            'temporal': temporal_weight / total,
            'degradation': degradation_weight / total
        }
    
    def fuse_scores(self,
                   motion_score: np.ndarray,
                   scale_score: np.ndarray,
                   temporal_score: np.ndarray,
                   degradation_weight: float) -> Dict:
        """Fuse all consistency scores into final anomaly map."""
        # Ensure all scores are same size
        h, w = motion_score.shape
        
        if scale_score.shape != (h, w):
            scale_score = cv2.resize(scale_score, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if temporal_score.shape != (h, w):
            temporal_score = cv2.resize(temporal_score, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize all scores to [0, 1]
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 0:
                return (x - x_min) / (x_max - x_min)
            return x
        
        motion_norm = normalize(motion_score)
        scale_norm = normalize(scale_score)
        temporal_norm = normalize(temporal_score)
        
        # Weighted fusion
        anomaly_score = (
            self.weights['motion'] * motion_norm +
            self.weights['scale'] * scale_norm +
            self.weights['temporal'] * temporal_norm
        )
        
        # Apply degradation consistency as multiplicative weight
        anomaly_score = anomaly_score * (degradation_weight ** self.weights['degradation'])
        
        # Final normalization
        anomaly_score = normalize(anomaly_score)
        
        # Compute confidence based on agreement between cues
        scores_stack = np.stack([motion_norm, scale_norm, temporal_norm], axis=0)
        agreement = 1.0 - np.std(scores_stack, axis=0)
        
        confidence = agreement * anomaly_score
        
        return {
            'anomaly_score': anomaly_score,
            'confidence': confidence,
            'normalized_scores': {
                'motion': motion_norm,
                'scale': scale_norm,
                'temporal': temporal_norm
            }
        }
    
    def generate_visualization(self,
                              frame: np.ndarray,
                              anomaly_score: np.ndarray,
                              confidence: np.ndarray,
                              threshold: float = 0.5) -> np.ndarray:
        """Generate visualization overlay."""
        # Create RGB visualization
        viz = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Create heatmap overlay
        heatmap = cm.jet(anomaly_score)[:, :, :3]  # RGB only
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Blend with confidence as alpha
        alpha = (confidence * 0.7).clip(0, 1)
        alpha = np.expand_dims(alpha, axis=2)
        
        overlay = (viz * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        
        # Draw contours for high-confidence regions
        binary = (anomaly_score > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        return overlay
