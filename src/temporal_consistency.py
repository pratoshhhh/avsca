"""
Stage 5: Temporal Consistency Analysis

Analyzes temporal persistence of anomalies across frames.

Rationale: True anomalies persist over multiple frames. Transient noise
and artifacts appear and disappear randomly.
"""

import numpy as np
from collections import deque
from typing import List, Dict


class TemporalConsistencyAnalyzer:
    """
    Analyzes temporal persistence of anomalies across frames.
    
    Rationale: True anomalies persist over multiple frames. Transient noise
    and artifacts appear and disappear randomly.
    """
    
    def __init__(self, window_size: int = 5, min_persistence: float = 0.6):
        """
        Args:
            window_size: Number of frames to consider for temporal consistency
            min_persistence: Minimum temporal consistency score to keep detection
        """
        self.window_size = window_size
        self.min_persistence = min_persistence
        self.history = deque(maxlen=window_size)
    
    def compute_temporal_coherence(self, 
                                  current_map: np.ndarray,
                                  history_maps: List[np.ndarray]) -> np.ndarray:
        """Compute pixel-wise temporal coherence score."""
        if len(history_maps) == 0:
            return np.ones_like(current_map)
        
        # Stack history
        history_stack = np.stack(history_maps, axis=0)
        
        # Compute temporal variance (low variance = high consistency)
        temporal_mean = np.mean(history_stack, axis=0)
        temporal_var = np.var(history_stack, axis=0)
        
        # Coherence score: high if current matches mean and variance is low
        agreement = 1.0 - np.abs(current_map - temporal_mean)
        stability = 1.0 / (1.0 + temporal_var)
        
        coherence = agreement * stability
        
        return coherence
    
    def compute_spatial_overlap(self,
                               current_mask: np.ndarray,
                               history_masks: List[np.ndarray]) -> float:
        """Compute IoU-based spatial overlap with history."""
        if len(history_masks) == 0:
            return 0.0
        
        overlaps = []
        for hist_mask in history_masks:
            intersection = np.logical_and(current_mask, hist_mask).sum()
            union = np.logical_or(current_mask, hist_mask).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            overlaps.append(iou)
        
        return np.mean(overlaps)
    
    def update_and_analyze(self, 
                          inconsistency_map: np.ndarray,
                          binary_mask: np.ndarray) -> Dict:
        """Update temporal history and compute consistency."""
        # Compute temporal coherence with history
        history_maps = [h['map'] for h in self.history]
        coherence = self.compute_temporal_coherence(inconsistency_map, history_maps)
        
        # Compute spatial overlap
        history_masks = [h['mask'] for h in self.history]
        spatial_overlap = self.compute_spatial_overlap(binary_mask, history_masks)
        
        # Combined temporal consistency score
        temporal_score = coherence * inconsistency_map
        
        # Update history
        self.history.append({
            'map': inconsistency_map,
            'mask': binary_mask
        })
        
        # Persistence-filtered mask
        persistence_mask = (temporal_score > self.min_persistence).astype(np.uint8)
        
        return {
            'temporal_coherence': coherence,
            'temporal_score': temporal_score,
            'spatial_overlap': spatial_overlap,
            'persistence_mask': persistence_mask,
            'num_history_frames': len(self.history)
        }
