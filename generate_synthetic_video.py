#!/usr/bin/env python3
"""
Synthetic Video Generator for Airborne Anomaly Detection Testing

This script generates synthetic aerial/airborne video footage with controllable
anomalies for testing the anomaly detection pipeline.

Features:
- Simulated aerial terrain background with camera motion
- Controllable moving objects (anomalies)
- Configurable noise, blur, and compression artifacts
- Ground truth anomaly masks for evaluation

Author: Defense-Oriented Computer Vision Lab
Date: January 2026
"""

import numpy as np
import cv2
import argparse
from typing import Tuple, List, Optional
import os


class TerrainGenerator:
    """Generates synthetic terrain-like background textures."""
    
    def __init__(self, width: int = 640, height: int = 480, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)
        
    def generate_perlin_noise(self, scale: float = 100.0) -> np.ndarray:
        """Generate Perlin-like noise for terrain texture."""
        # Create base noise at different frequencies
        noise = np.zeros((self.height, self.width), dtype=np.float32)
        
        frequencies = [1, 2, 4, 8, 16]
        amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]
        
        for freq, amp in zip(frequencies, amplitudes):
            # Generate random grid
            divisor = max(1, int(scale / freq))
            grid_h = max(2, self.height // divisor)
            grid_w = max(2, self.width // divisor)
            
            small_noise = self.rng.rand(grid_h, grid_w).astype(np.float32)
            
            # Upscale with interpolation
            upscaled = cv2.resize(small_noise, (self.width, self.height), 
                                 interpolation=cv2.INTER_CUBIC)
            
            noise += upscaled * amp
        
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        
        return noise
    
    def generate_terrain(self, terrain_type: str = 'mixed') -> np.ndarray:
        """
        Generate terrain texture.
        
        Args:
            terrain_type: 'urban', 'rural', 'desert', 'forest', 'mixed'
        """
        base_noise = self.generate_perlin_noise(scale=50.0)
        detail_noise = self.generate_perlin_noise(scale=10.0)
        
        # Combine different noise scales
        terrain = 0.7 * base_noise + 0.3 * detail_noise
        
        # Apply color/intensity based on terrain type
        if terrain_type == 'urban':
            # More structured, grid-like patterns
            grid = np.zeros_like(terrain)
            block_size = 40
            for y in range(0, self.height, block_size):
                for x in range(0, self.width, block_size):
                    intensity = self.rng.uniform(0.2, 0.8)
                    y_end = min(y + block_size - 2, self.height)
                    x_end = min(x + block_size - 2, self.width)
                    grid[y:y_end, x:x_end] = intensity
            terrain = 0.5 * terrain + 0.5 * grid
            
        elif terrain_type == 'desert':
            # More uniform, sandy colors
            terrain = 0.3 + 0.4 * terrain
            
        elif terrain_type == 'forest':
            # Darker, more varied
            terrain = 0.1 + 0.5 * terrain
            
        elif terrain_type == 'rural':
            # Field-like patterns
            field_pattern = np.zeros_like(terrain)
            stripe_width = 30
            for i, y in enumerate(range(0, self.height, stripe_width)):
                intensity = 0.3 + 0.4 * (i % 2)
                y_end = min(y + stripe_width, self.height)
                field_pattern[y:y_end, :] = intensity
            terrain = 0.6 * terrain + 0.4 * field_pattern
        
        # Clip to valid range
        terrain = np.clip(terrain, 0, 1)
        
        return terrain


class CameraMotionSimulator:
    """Simulates camera motion for aerial footage."""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.position = np.array([0.0, 0.0])  # x, y position
        self.velocity = np.array([2.0, 1.0])  # pixels per frame
        self.rotation = 0.0  # degrees
        self.rotation_velocity = 0.1  # degrees per frame
        
    def set_motion_parameters(self, 
                             velocity: Tuple[float, float] = (2.0, 1.0),
                             rotation_velocity: float = 0.1,
                             motion_type: str = 'linear'):
        """Configure camera motion parameters."""
        self.velocity = np.array(velocity)
        self.rotation_velocity = rotation_velocity
        self.motion_type = motion_type
        
    def get_transform(self, frame_idx: int) -> np.ndarray:
        """Get affine transform for current frame."""
        if self.motion_type == 'linear':
            # Simple linear motion
            dx = self.velocity[0] * frame_idx
            dy = self.velocity[1] * frame_idx
            angle = self.rotation + self.rotation_velocity * frame_idx
            
        elif self.motion_type == 'sinusoidal':
            # Oscillating motion (like turbulence)
            dx = self.velocity[0] * frame_idx + 10 * np.sin(frame_idx * 0.1)
            dy = self.velocity[1] * frame_idx + 5 * np.cos(frame_idx * 0.15)
            angle = self.rotation + 2 * np.sin(frame_idx * 0.05)
            
        elif self.motion_type == 'circular':
            # Circular/orbit motion
            radius = 50
            dx = radius * np.cos(frame_idx * 0.05)
            dy = radius * np.sin(frame_idx * 0.05)
            angle = frame_idx * 0.5
            
        else:
            dx, dy, angle = 0, 0, 0
        
        # Build affine transform matrix
        center = (self.width / 2, self.height / 2)
        angle_rad = np.radians(angle)
        
        # Rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Affine transform: rotate around center, then translate
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * center[0] + sin_a * center[1] + dx],
            [sin_a, cos_a, -sin_a * center[0] + (1 - cos_a) * center[1] + dy]
        ], dtype=np.float32)
        
        return M


class AnomalyGenerator:
    """Generates moving anomaly objects."""
    
    def __init__(self, width: int = 640, height: int = 480, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)
        self.anomalies = []
        
    def add_anomaly(self, 
                   anomaly_type: str = 'ufo',
                   start_pos: Tuple[int, int] = None,
                   velocity: Tuple[float, float] = None,
                   size: int = 20,
                   intensity: float = 0.9):
        """
        Add an anomaly to the scene.
        
        Args:
            anomaly_type: 'ufo', 'aircraft', 'bird', 'debris'
            start_pos: Starting (x, y) position
            velocity: (vx, vy) velocity in pixels/frame
            size: Object size in pixels
            intensity: Brightness (0-1)
        """
        if start_pos is None:
            start_pos = (self.rng.randint(50, self.width - 50),
                        self.rng.randint(50, self.height - 50))
        
        if velocity is None:
            velocity = (self.rng.uniform(-3, 3), self.rng.uniform(-3, 3))
        
        self.anomalies.append({
            'type': anomaly_type,
            'position': np.array(start_pos, dtype=np.float32),
            'velocity': np.array(velocity, dtype=np.float32),
            'size': size,
            'intensity': intensity,
            'active': True
        })
    
    def get_anomaly_mask(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get anomaly rendering and ground truth mask for current frame.
        
        Returns:
            anomaly_layer: Grayscale anomaly image
            gt_mask: Binary ground truth mask
        """
        anomaly_layer = np.zeros((self.height, self.width), dtype=np.float32)
        gt_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for anomaly in self.anomalies:
            if not anomaly['active']:
                continue
            
            # Update position
            pos = anomaly['position'] + anomaly['velocity'] * frame_idx
            
            # Check bounds
            if (pos[0] < -anomaly['size'] or pos[0] > self.width + anomaly['size'] or
                pos[1] < -anomaly['size'] or pos[1] > self.height + anomaly['size']):
                continue
            
            x, y = int(pos[0]), int(pos[1])
            size = anomaly['size']
            intensity = anomaly['intensity']
            
            # Draw anomaly based on type
            if anomaly['type'] == 'ufo':
                # Elliptical bright object
                cv2.ellipse(anomaly_layer, (x, y), (size, size // 2), 
                           0, 0, 360, intensity, -1)
                cv2.ellipse(gt_mask, (x, y), (size, size // 2), 
                           0, 0, 360, 255, -1)
                
            elif anomaly['type'] == 'aircraft':
                # Cross-shaped (simplified aircraft)
                cv2.line(anomaly_layer, (x - size, y), (x + size, y), 
                        intensity, 3)
                cv2.line(anomaly_layer, (x, y - size // 2), (x, y + size // 2), 
                        intensity, 2)
                cv2.line(gt_mask, (x - size, y), (x + size, y), 255, 5)
                cv2.line(gt_mask, (x, y - size // 2), (x, y + size // 2), 255, 4)
                
            elif anomaly['type'] == 'bird':
                # Small moving dot
                cv2.circle(anomaly_layer, (x, y), size // 3, intensity, -1)
                cv2.circle(gt_mask, (x, y), size // 3, 255, -1)
                
            elif anomaly['type'] == 'debris':
                # Irregular shape
                pts = np.array([
                    [x + self.rng.randint(-size, size), 
                     y + self.rng.randint(-size, size)]
                    for _ in range(5)
                ], dtype=np.int32)
                cv2.fillPoly(anomaly_layer, [pts], intensity)
                cv2.fillPoly(gt_mask, [pts], 255)
        
        return anomaly_layer, gt_mask


class SyntheticVideoGenerator:
    """Main class for generating synthetic airborne video."""
    
    def __init__(self, 
                 width: int = 640, 
                 height: int = 480,
                 fps: float = 30.0,
                 seed: int = 42):
        self.width = width
        self.height = height
        self.fps = fps
        self.seed = seed
        
        self.terrain_gen = TerrainGenerator(width * 3, height * 3, seed)
        self.camera = CameraMotionSimulator(width, height)
        self.anomaly_gen = AnomalyGenerator(width, height, seed)
        
        # Generate large terrain (3x size for camera motion)
        self.terrain = None
        
    def setup_scene(self,
                   terrain_type: str = 'mixed',
                   motion_type: str = 'linear',
                   motion_velocity: Tuple[float, float] = (2.0, 1.0),
                   num_anomalies: int = 2):
        """Configure the synthetic scene."""
        print(f"Generating {terrain_type} terrain...")
        self.terrain = self.terrain_gen.generate_terrain(terrain_type)
        
        print(f"Setting up {motion_type} camera motion...")
        self.camera.set_motion_parameters(
            velocity=motion_velocity,
            motion_type=motion_type
        )
        
        print(f"Adding {num_anomalies} anomalies...")
        anomaly_types = ['ufo', 'aircraft', 'bird', 'debris']
        for i in range(num_anomalies):
            atype = anomaly_types[i % len(anomaly_types)]
            self.anomaly_gen.add_anomaly(
                anomaly_type=atype,
                size=15 + i * 5
            )
    
    def add_noise(self, frame: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to frame."""
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.float32)
        noisy = frame + noise
        return np.clip(noisy, 0, 1)
    
    def add_blur(self, frame: np.ndarray, blur_amount: float = 1.0) -> np.ndarray:
        """Add motion blur to frame."""
        if blur_amount <= 0:
            return frame
        
        ksize = int(blur_amount * 5) | 1  # Ensure odd
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_amount)
        return blurred
    
    def generate_frame(self, frame_idx: int, 
                      noise_level: float = 0.02,
                      blur_amount: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single frame.
        
        Returns:
            frame: Generated grayscale frame [0, 1]
            gt_mask: Ground truth anomaly mask
        """
        # Get camera transform
        M = self.camera.get_transform(frame_idx)
        
        # Extract view from large terrain
        # Center the view in the larger terrain
        offset_x = self.terrain.shape[1] // 3
        offset_y = self.terrain.shape[0] // 3
        
        # Adjust transform for offset
        M_adjusted = M.copy()
        M_adjusted[0, 2] += offset_x
        M_adjusted[1, 2] += offset_y
        
        # Warp terrain to simulate camera motion
        background = cv2.warpAffine(
            self.terrain, M_adjusted, 
            (self.width, self.height),
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Get anomaly layer and mask
        anomaly_layer, gt_mask = self.anomaly_gen.get_anomaly_mask(frame_idx)
        
        # Composite anomalies onto background
        frame = np.where(anomaly_layer > 0.1, anomaly_layer, background)
        
        # Add degradation
        frame = self.add_blur(frame, blur_amount)
        frame = self.add_noise(frame, noise_level)
        
        return frame.astype(np.float32), gt_mask
    
    def generate_video(self,
                      output_path: str,
                      num_frames: int = 100,
                      noise_level: float = 0.02,
                      blur_amount: float = 0.5,
                      save_ground_truth: bool = True) -> str:
        """
        Generate complete synthetic video.
        
        Args:
            output_path: Output video file path
            num_frames: Number of frames to generate
            noise_level: Amount of Gaussian noise (0-1)
            blur_amount: Amount of blur (0-5)
            save_ground_truth: Whether to save ground truth masks
            
        Returns:
            Path to generated video
        """
        print(f"\nGenerating {num_frames} frames...")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height), isColor=False)
        
        # Setup ground truth writer if requested
        gt_out = None
        if save_ground_truth:
            gt_path = output_path.replace('.mp4', '_groundtruth.mp4')
            gt_out = cv2.VideoWriter(gt_path, fourcc, self.fps,
                                    (self.width, self.height), isColor=False)
        
        for i in range(num_frames):
            frame, gt_mask = self.generate_frame(i, noise_level, blur_amount)
            
            # Convert to uint8 for video writing
            frame_uint8 = (frame * 255).astype(np.uint8)
            out.write(frame_uint8)
            
            if gt_out is not None:
                gt_out.write(gt_mask)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{num_frames} frames")
        
        out.release()
        if gt_out is not None:
            gt_out.release()
            print(f"Ground truth saved to: {gt_path}")
        
        print(f"Video saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic airborne video for anomaly detection testing'
    )
    parser.add_argument('-o', '--output', type=str, default='synthetic_aerial.mp4',
                       help='Output video path')
    parser.add_argument('-n', '--num-frames', type=int, default=100,
                       help='Number of frames to generate')
    parser.add_argument('-W', '--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('-H', '--height', type=int, default=480,
                       help='Video height')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frames per second')
    parser.add_argument('--terrain', type=str, default='mixed',
                       choices=['urban', 'rural', 'desert', 'forest', 'mixed'],
                       help='Terrain type')
    parser.add_argument('--motion', type=str, default='linear',
                       choices=['linear', 'sinusoidal', 'circular'],
                       help='Camera motion type')
    parser.add_argument('--anomalies', type=int, default=2,
                       help='Number of anomaly objects')
    parser.add_argument('--noise', type=float, default=0.02,
                       help='Noise level (0-1)')
    parser.add_argument('--blur', type=float, default=0.5,
                       help='Blur amount (0-5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-groundtruth', action='store_true',
                       help='Skip saving ground truth masks')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNTHETIC AIRBORNE VIDEO GENERATOR")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Output: {args.output}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames} @ {args.fps} FPS")
    print(f"  Terrain: {args.terrain}")
    print(f"  Motion: {args.motion}")
    print(f"  Anomalies: {args.anomalies}")
    print(f"  Noise: {args.noise}")
    print(f"  Blur: {args.blur}")
    
    # Create generator
    generator = SyntheticVideoGenerator(
        width=args.width,
        height=args.height,
        fps=args.fps,
        seed=args.seed
    )
    
    # Setup scene
    generator.setup_scene(
        terrain_type=args.terrain,
        motion_type=args.motion,
        num_anomalies=args.anomalies
    )
    
    # Generate video
    generator.generate_video(
        output_path=args.output,
        num_frames=args.num_frames,
        noise_level=args.noise,
        blur_amount=args.blur,
        save_ground_truth=not args.no_groundtruth
    )
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nTo run anomaly detection on this video:")
    print(f"  python main.py")
    print(f"  # Then update VIDEO_PATH = '{args.output}'")


if __name__ == "__main__":
    main()
