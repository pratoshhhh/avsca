"""
Stage 2: Global Background Motion Estimation

Estimates global background motion between frames using optical flow and RANSAC.

Rationale: Camera motion from the airborne platform creates a dominant
background motion field. Anomalies violate this pattern.
"""

import numpy as np
import cv2
from typing import Tuple, Dict


class BackgroundMotionEstimator:
    """
    Estimates global background motion between frames using optical flow and RANSAC.
    
    Rationale: Camera motion from the airborne platform creates a dominant
    background motion field. Anomalies violate this pattern.
    """
    
    def __init__(self, model_type: str = 'affine'):
        """
        Args:
            model_type: 'affine' or 'homography'
        """
        self.model_type = model_type
        
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow using Farnebäck algorithm.
        Returns flow field of shape (H, W, 2) with (dx, dy) at each pixel.
        """
        # Ensure uint8 format for OpenCV
        img1 = (frame1 * 255).astype(np.uint8)
        img2 = (frame2 * 255).astype(np.uint8)
        
        # Farnebäck optical flow
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2,
            flow=None,
            pyr_scale=0.5,      # Pyramid scale
            levels=3,            # Number of pyramid levels
            winsize=15,          # Window size
            iterations=3,        # Iterations at each level
            poly_n=5,            # Polynomial expansion neighborhood
            poly_sigma=1.2,      # Gaussian sigma for polynomial expansion
            flags=0
        )
        
        return flow
    
    def estimate_global_motion_ransac(self, flow: np.ndarray, 
                                     inlier_threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit global motion model using RANSAC.
        
        Returns:
            transform_matrix: Affine or homography matrix
            inlier_mask: Boolean mask of inliers
        """
        h, w = flow.shape[:2]
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Sample points (subsample for efficiency)
        step = 10
        y_sample = y_coords[::step, ::step].flatten()
        x_sample = x_coords[::step, ::step].flatten()
        flow_sample = flow[::step, ::step].reshape(-1, 2)
        
        # Source points (current positions)
        src_pts = np.column_stack([x_sample, y_sample]).astype(np.float32)
        
        # Destination points (predicted positions)
        dst_pts = src_pts + flow_sample
        
        # Remove invalid flow vectors
        valid_mask = ~(np.isnan(flow_sample).any(axis=1) | np.isinf(flow_sample).any(axis=1))
        src_pts = src_pts[valid_mask]
        dst_pts = dst_pts[valid_mask]
        
        if len(src_pts) < 10:
            # Not enough points, return identity
            if self.model_type == 'affine':
                return np.eye(2, 3, dtype=np.float32), np.zeros(len(src_pts), dtype=bool)
            else:
                return np.eye(3, dtype=np.float32), np.zeros(len(src_pts), dtype=bool)
        
        # RANSAC estimation
        if self.model_type == 'affine':
            transform, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=inlier_threshold,
                maxIters=2000,
                confidence=0.99
            )
            if transform is None:
                transform = np.eye(2, 3, dtype=np.float32)
                inliers = np.zeros(len(src_pts), dtype=bool)
        else:
            transform, inliers = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=inlier_threshold,
                maxIters=2000,
                confidence=0.99
            )
            if transform is None:
                transform = np.eye(3, dtype=np.float32)
                inliers = np.zeros(len(src_pts), dtype=bool)
        
        inliers = inliers.flatten().astype(bool)
        
        return transform, inliers
    
    def predict_flow_from_transform(self, transform: np.ndarray, 
                                   shape: Tuple[int, int]) -> np.ndarray:
        """Generate predicted flow field from global motion transform."""
        h, w = shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Flatten coordinates
        coords = np.column_stack([x_coords.flatten(), y_coords.flatten()]).astype(np.float32)
        
        # Apply transform
        if self.model_type == 'affine':
            # Affine: [x', y'] = M @ [x, y, 1]
            coords_h = np.column_stack([coords, np.ones(len(coords))])
            predicted = (transform @ coords_h.T).T
        else:
            # Homography
            coords_h = np.column_stack([coords, np.ones(len(coords))])
            predicted_h = (transform @ coords_h.T).T
            predicted = predicted_h[:, :2] / predicted_h[:, 2:3]
        
        # Compute flow
        flow_flat = predicted - coords
        predicted_flow = flow_flat.reshape(h, w, 2)
        
        return predicted_flow
    
    def process_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict:
        """Process a frame pair and return flow analysis results."""
        # Compute optical flow
        flow = self.compute_optical_flow(frame1, frame2)
        
        # Estimate global motion
        transform, inliers = self.estimate_global_motion_ransac(flow)
        
        # Predict background flow
        predicted_flow = self.predict_flow_from_transform(transform, frame1.shape)
        
        # Compute residual
        residual_flow = flow - predicted_flow
        residual_magnitude = np.sqrt(residual_flow[:, :, 0]**2 + residual_flow[:, :, 1]**2)
        
        return {
            'flow': flow,
            'transform': transform,
            'predicted_flow': predicted_flow,
            'residual_flow': residual_flow,
            'residual_magnitude': residual_magnitude,
            'inlier_ratio': inliers.sum() / len(inliers) if len(inliers) > 0 else 0
        }
