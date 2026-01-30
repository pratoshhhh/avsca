"""
Stage 1: Image Quality & Physical Degradation Estimation

Estimates physical degradation metrics for each frame.

Rationale: Physical degradation (blur, noise, contrast loss) provides
critical context. Anomalies should degrade consistently with the rest
of the image.
"""

import numpy as np
import cv2
from scipy import signal
from typing import List, Dict


class ImageDegradationEstimator:
    """
    Estimates physical degradation metrics for each frame.
    
    Rationale: Physical degradation (blur, noise, contrast loss) provides
    critical context. Anomalies should degrade consistently with the rest
    of the image.
    """
    
    @staticmethod
    def estimate_motion_blur(image: np.ndarray) -> float:
        """
        Estimate motion blur using frequency domain analysis.
        Higher values indicate more blur.
        
        Method: Compute ratio of high-frequency to low-frequency energy.
        Blurred images have less high-frequency content.
        """
        # Compute 2D FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Create frequency radius map
        h, w = image.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Define frequency bands
        max_radius = min(h, w) // 2
        low_freq_mask = radius < (max_radius * 0.15)
        high_freq_mask = radius > (max_radius * 0.5)
        
        # Compute energy ratio
        low_energy = np.sum(magnitude[low_freq_mask])
        high_energy = np.sum(magnitude[high_freq_mask])
        
        # Blur metric: lower ratio = more blur
        if low_energy > 0:
            blur_metric = 1.0 - (high_energy / (low_energy + high_energy))
        else:
            blur_metric = 1.0
        
        return blur_metric
    
    @staticmethod
    def estimate_noise_level(image: np.ndarray) -> float:
        """
        Estimate noise level using MAD (Median Absolute Deviation) method.
        
        Method: Apply high-pass filter and compute MAD of result.
        Assumes noise is independent high-frequency content.
        """
        # Convert to uint8 for Laplacian, then back to float
        image_uint8 = (image * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F, ksize=3)
        
        # Compute MAD (robust to outliers)
        median = np.median(laplacian)
        mad = np.median(np.abs(laplacian - median))
        
        # Scale to standard deviation estimate
        noise_sigma = 1.4826 * mad
        
        return noise_sigma
    
    @staticmethod
    def estimate_contrast(image: np.ndarray, window_size: int = 32) -> float:
        """
        Estimate global contrast using local standard deviation.
        
        Method: Compute mean of local standard deviations.
        Higher values indicate better contrast.
        """
        # Compute local standard deviation using sliding window
        mean_filter = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = signal.convolve2d(image, mean_filter, mode='same', boundary='symm')
        local_mean_sq = signal.convolve2d(image**2, mean_filter, mode='same', boundary='symm')
        local_std = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0))
        
        # Global contrast is mean of local std devs
        contrast = np.mean(local_std)
        
        return contrast
    
    def process_sequence(self, frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Process entire frame sequence and return degradation metrics."""
        n_frames = len(frames)
        blur_scores = np.zeros(n_frames)
        noise_scores = np.zeros(n_frames)
        contrast_scores = np.zeros(n_frames)
        
        print("Estimating image degradation metrics...")
        for i, frame in enumerate(frames):
            blur_scores[i] = self.estimate_motion_blur(frame)
            noise_scores[i] = self.estimate_noise_level(frame)
            contrast_scores[i] = self.estimate_contrast(frame)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{n_frames} frames")
        
        return {
            'blur': blur_scores,
            'noise': noise_scores,
            'contrast': contrast_scores
        }
