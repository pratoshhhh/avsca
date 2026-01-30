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
- Multiple output formats and export options

Author: Defense-Oriented Computer Vision Lab
Date: January 2026
"""

import numpy as np
import cv2
import argparse
from typing import Tuple, List, Optional, Dict
import os
import json


class TerrainGenerator:
    """Generates synthetic terrain-like background textures."""
    
    def __init__(self, width: int = 640, height: int = 480, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)
        
    def generate_perlin_noise(self, scale: float = 100.0) -> np.ndarray:
        """Generate Perlin-like noise for terrain texture."""
        noise = np.zeros((self.height, self.width), dtype=np.float32)
        
        frequencies = [1, 2, 4, 8, 16]
        amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]
        
        for freq, amp in zip(frequencies, amplitudes):
            divisor = max(1, int(scale / freq))
            grid_h = max(2, self.height // divisor)
            grid_w = max(2, self.width // divisor)
            
            small_noise = self.rng.rand(grid_h, grid_w).astype(np.float32)
            upscaled = cv2.resize(small_noise, (self.width, self.height), 
                                 interpolation=cv2.INTER_CUBIC)
            noise += upscaled * amp
        
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        return noise
    
    def generate_terrain(self, terrain_type: str = 'mixed', 
                        texture_scale: float = 1.0,
                        contrast: float = 1.0) -> np.ndarray:
        """
        Generate terrain texture.
        
        Args:
            terrain_type: 'urban', 'rural', 'desert', 'forest', 'mixed', 'water', 'mountain'
            texture_scale: Scale factor for texture detail (0.5-2.0)
            contrast: Contrast adjustment (0.5-2.0)
        """
        base_noise = self.generate_perlin_noise(scale=50.0 * texture_scale)
        detail_noise = self.generate_perlin_noise(scale=10.0 * texture_scale)
        
        terrain = 0.7 * base_noise + 0.3 * detail_noise
        
        if terrain_type == 'urban':
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
            terrain = 0.3 + 0.4 * terrain
            
        elif terrain_type == 'forest':
            terrain = 0.1 + 0.5 * terrain
            
        elif terrain_type == 'rural':
            field_pattern = np.zeros_like(terrain)
            stripe_width = 30
            for i, y in enumerate(range(0, self.height, stripe_width)):
                intensity = 0.3 + 0.4 * (i % 2)
                y_end = min(y + stripe_width, self.height)
                field_pattern[y:y_end, :] = intensity
            terrain = 0.6 * terrain + 0.4 * field_pattern
            
        elif terrain_type == 'water':
            # Wavy water-like patterns
            x = np.linspace(0, 4 * np.pi, self.width)
            y = np.linspace(0, 4 * np.pi, self.height)
            X, Y = np.meshgrid(x, y)
            waves = 0.5 + 0.2 * np.sin(X + Y) + 0.1 * np.sin(2*X - Y)
            terrain = 0.3 * terrain + 0.7 * waves.astype(np.float32)
            terrain = 0.2 + 0.3 * terrain  # Darker for water
            
        elif terrain_type == 'mountain':
            # Ridged noise for mountains
            ridge_noise = np.abs(terrain - 0.5) * 2
            terrain = 0.4 * terrain + 0.6 * ridge_noise
        
        # Apply contrast
        terrain = (terrain - 0.5) * contrast + 0.5
        terrain = np.clip(terrain, 0, 1)
        
        return terrain


class CameraMotionSimulator:
    """Simulates camera motion for aerial footage."""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([2.0, 1.0])
        self.rotation = 0.0
        self.rotation_velocity = 0.1
        self.zoom_factor = 1.0
        self.zoom_velocity = 0.0
        
    def set_motion_parameters(self, 
                             velocity: Tuple[float, float] = (2.0, 1.0),
                             rotation_velocity: float = 0.1,
                             motion_type: str = 'linear',
                             zoom_velocity: float = 0.0,
                             turbulence: float = 0.0):
        """Configure camera motion parameters."""
        self.velocity = np.array(velocity)
        self.rotation_velocity = rotation_velocity
        self.motion_type = motion_type
        self.zoom_velocity = zoom_velocity
        self.turbulence = turbulence
        
    def get_transform(self, frame_idx: int) -> np.ndarray:
        """Get affine transform for current frame."""
        # Add turbulence (random shake)
        turb_x = self.turbulence * np.random.randn() if self.turbulence > 0 else 0
        turb_y = self.turbulence * np.random.randn() if self.turbulence > 0 else 0
        
        if self.motion_type == 'linear':
            dx = self.velocity[0] * frame_idx + turb_x
            dy = self.velocity[1] * frame_idx + turb_y
            angle = self.rotation + self.rotation_velocity * frame_idx
            
        elif self.motion_type == 'sinusoidal':
            dx = self.velocity[0] * frame_idx + 10 * np.sin(frame_idx * 0.1) + turb_x
            dy = self.velocity[1] * frame_idx + 5 * np.cos(frame_idx * 0.15) + turb_y
            angle = self.rotation + 2 * np.sin(frame_idx * 0.05)
            
        elif self.motion_type == 'circular':
            radius = 50
            dx = radius * np.cos(frame_idx * 0.05) + turb_x
            dy = radius * np.sin(frame_idx * 0.05) + turb_y
            angle = frame_idx * 0.5
            
        elif self.motion_type == 'hover':
            # Minimal movement, like hovering drone
            dx = 2 * np.sin(frame_idx * 0.02) + turb_x
            dy = 2 * np.cos(frame_idx * 0.03) + turb_y
            angle = 0.5 * np.sin(frame_idx * 0.01)
            
        elif self.motion_type == 'dive':
            # Accelerating forward motion (diving)
            t = frame_idx / 30.0
            dx = self.velocity[0] * t * t + turb_x
            dy = self.velocity[1] * t + turb_y
            angle = -t * 2  # Tilting down
            
        elif self.motion_type == 'sweep':
            # Side-to-side sweeping
            dx = 30 * np.sin(frame_idx * 0.05) + turb_x
            dy = self.velocity[1] * frame_idx + turb_y
            angle = 5 * np.sin(frame_idx * 0.05)
            
        else:
            dx, dy, angle = turb_x, turb_y, 0
        
        # Apply zoom
        zoom = 1.0 + self.zoom_velocity * frame_idx
        zoom = max(0.5, min(2.0, zoom))  # Clamp zoom
        
        center = (self.width / 2, self.height / 2)
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Affine transform with zoom
        M = np.array([
            [cos_a * zoom, -sin_a * zoom, (1 - cos_a * zoom) * center[0] + sin_a * zoom * center[1] + dx],
            [sin_a * zoom, cos_a * zoom, -sin_a * zoom * center[0] + (1 - cos_a * zoom) * center[1] + dy]
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
                   intensity: float = 0.9,
                   acceleration: Tuple[float, float] = (0, 0),
                   rotation_speed: float = 0,
                   pulsing: bool = False,
                   trail: bool = False):
        """
        Add an anomaly to the scene.
        
        Args:
            anomaly_type: 'ufo', 'aircraft', 'bird', 'debris', 'drone', 'missile', 'balloon'
            start_pos: Starting (x, y) position
            velocity: (vx, vy) velocity in pixels/frame
            size: Object size in pixels
            intensity: Brightness (0-1)
            acceleration: (ax, ay) acceleration in pixels/frame^2
            rotation_speed: Object rotation speed in degrees/frame
            pulsing: Whether object brightness pulses
            trail: Whether to leave a motion trail
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
            'acceleration': np.array(acceleration, dtype=np.float32),
            'size': size,
            'intensity': intensity,
            'rotation': 0,
            'rotation_speed': rotation_speed,
            'pulsing': pulsing,
            'trail': trail,
            'trail_positions': [],
            'active': True
        })
    
    def get_anomaly_mask(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly rendering and ground truth mask for current frame."""
        anomaly_layer = np.zeros((self.height, self.width), dtype=np.float32)
        gt_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for anomaly in self.anomalies:
            if not anomaly['active']:
                continue
            
            # Update position with acceleration
            t = frame_idx
            pos = (anomaly['position'] + 
                   anomaly['velocity'] * t + 
                   0.5 * anomaly['acceleration'] * t * t)
            
            # Check bounds
            if (pos[0] < -anomaly['size'] or pos[0] > self.width + anomaly['size'] or
                pos[1] < -anomaly['size'] or pos[1] > self.height + anomaly['size']):
                continue
            
            x, y = int(pos[0]), int(pos[1])
            size = anomaly['size']
            
            # Handle pulsing intensity
            if anomaly['pulsing']:
                intensity = anomaly['intensity'] * (0.7 + 0.3 * np.sin(frame_idx * 0.3))
            else:
                intensity = anomaly['intensity']
            
            # Draw trail if enabled
            if anomaly['trail'] and len(anomaly['trail_positions']) > 0:
                for i, (tx, ty) in enumerate(anomaly['trail_positions'][-10:]):
                    trail_alpha = (i + 1) / 10 * 0.3
                    cv2.circle(anomaly_layer, (int(tx), int(ty)), 
                              max(1, size // 4), intensity * trail_alpha, -1)
            
            # Update trail
            if anomaly['trail']:
                anomaly['trail_positions'].append((x, y))
            
            # Calculate rotation
            rotation = anomaly['rotation'] + anomaly['rotation_speed'] * frame_idx
            
            # Draw anomaly based on type
            if anomaly['type'] == 'ufo':
                cv2.ellipse(anomaly_layer, (x, y), (size, size // 2), 
                           rotation, 0, 360, intensity, -1)
                cv2.ellipse(gt_mask, (x, y), (size, size // 2), 
                           rotation, 0, 360, 255, -1)
                
            elif anomaly['type'] == 'aircraft':
                cv2.line(anomaly_layer, (x - size, y), (x + size, y), intensity, 3)
                cv2.line(anomaly_layer, (x, y - size // 2), (x, y + size // 2), intensity, 2)
                cv2.line(gt_mask, (x - size, y), (x + size, y), 255, 5)
                cv2.line(gt_mask, (x, y - size // 2), (x, y + size // 2), 255, 4)
                
            elif anomaly['type'] == 'bird':
                cv2.circle(anomaly_layer, (x, y), size // 3, intensity, -1)
                cv2.circle(gt_mask, (x, y), size // 3, 255, -1)
                
            elif anomaly['type'] == 'debris':
                pts = np.array([
                    [x + self.rng.randint(-size, size), 
                     y + self.rng.randint(-size, size)]
                    for _ in range(5)
                ], dtype=np.int32)
                cv2.fillPoly(anomaly_layer, [pts], intensity)
                cv2.fillPoly(gt_mask, [pts], 255)
                
            elif anomaly['type'] == 'drone':
                # Quadcopter shape
                arm_len = size // 2
                for angle in [45, 135, 225, 315]:
                    rad = np.radians(angle + rotation)
                    ex = int(x + arm_len * np.cos(rad))
                    ey = int(y + arm_len * np.sin(rad))
                    cv2.line(anomaly_layer, (x, y), (ex, ey), intensity, 2)
                    cv2.circle(anomaly_layer, (ex, ey), size // 6, intensity, -1)
                    cv2.line(gt_mask, (x, y), (ex, ey), 255, 3)
                    cv2.circle(gt_mask, (ex, ey), size // 6, 255, -1)
                    
            elif anomaly['type'] == 'missile':
                # Elongated shape with trail
                length = size * 2
                rad = np.radians(rotation)
                ex = int(x + length * np.cos(rad))
                ey = int(y + length * np.sin(rad))
                cv2.line(anomaly_layer, (x, y), (ex, ey), intensity, 3)
                cv2.circle(anomaly_layer, (ex, ey), size // 4, intensity * 1.2, -1)
                cv2.line(gt_mask, (x, y), (ex, ey), 255, 5)
                cv2.circle(gt_mask, (ex, ey), size // 4, 255, -1)
                
            elif anomaly['type'] == 'balloon':
                # Round with string
                cv2.circle(anomaly_layer, (x, y), size, intensity, -1)
                cv2.line(anomaly_layer, (x, y + size), (x, y + size + size), 
                        intensity * 0.5, 1)
                cv2.circle(gt_mask, (x, y), size, 255, -1)
                cv2.line(gt_mask, (x, y + size), (x, y + size + size), 255, 2)
                
            elif anomaly['type'] == 'helicopter':
                # Body + rotor
                cv2.ellipse(anomaly_layer, (x, y), (size, size // 3), 0, 0, 360, intensity, -1)
                rotor_angle = (frame_idx * 30) % 360
                for da in [0, 90, 180, 270]:
                    rad = np.radians(rotor_angle + da)
                    rx = int(x + size * 1.5 * np.cos(rad))
                    ry = int(y + size * 1.5 * np.sin(rad))
                    cv2.line(anomaly_layer, (x, y), (rx, ry), intensity * 0.7, 1)
                cv2.ellipse(gt_mask, (x, y), (size, size // 3), 0, 0, 360, 255, -1)
        
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
        self.terrain = None
        self.config = {}
        
    def setup_scene(self,
                   terrain_type: str = 'mixed',
                   motion_type: str = 'linear',
                   motion_velocity: Tuple[float, float] = (2.0, 1.0),
                   rotation_velocity: float = 0.1,
                   zoom_velocity: float = 0.0,
                   turbulence: float = 0.0,
                   num_anomalies: int = 2,
                   anomaly_types: List[str] = None,
                   anomaly_speed: float = 1.0,
                   anomaly_size_range: Tuple[int, int] = (15, 40),
                   texture_scale: float = 1.0,
                   terrain_contrast: float = 1.0):
        """Configure the synthetic scene with extended parameters."""
        
        self.config = {
            'terrain_type': terrain_type,
            'motion_type': motion_type,
            'motion_velocity': motion_velocity,
            'rotation_velocity': rotation_velocity,
            'zoom_velocity': zoom_velocity,
            'turbulence': turbulence,
            'num_anomalies': num_anomalies,
            'anomaly_types': anomaly_types,
            'anomaly_speed': anomaly_speed,
            'anomaly_size_range': anomaly_size_range,
            'texture_scale': texture_scale,
            'terrain_contrast': terrain_contrast
        }
        
        print(f"Generating {terrain_type} terrain (scale={texture_scale}, contrast={terrain_contrast})...")
        self.terrain = self.terrain_gen.generate_terrain(
            terrain_type, 
            texture_scale=texture_scale,
            contrast=terrain_contrast
        )
        
        print(f"Setting up {motion_type} camera motion...")
        print(f"  Velocity: {motion_velocity}, Rotation: {rotation_velocity} deg/frame")
        print(f"  Zoom: {zoom_velocity}/frame, Turbulence: {turbulence}")
        self.camera.set_motion_parameters(
            velocity=motion_velocity,
            rotation_velocity=rotation_velocity,
            motion_type=motion_type,
            zoom_velocity=zoom_velocity,
            turbulence=turbulence
        )
        
        # Default anomaly types
        if anomaly_types is None:
            anomaly_types = ['ufo', 'aircraft', 'bird', 'debris', 'drone', 'missile', 'balloon', 'helicopter']
        
        print(f"Adding {num_anomalies} anomalies (speed={anomaly_speed}x)...")
        print(f"  Types: {anomaly_types}")
        print(f"  Size range: {anomaly_size_range}")
        
        for i in range(num_anomalies):
            atype = anomaly_types[i % len(anomaly_types)]
            size = np.random.randint(anomaly_size_range[0], anomaly_size_range[1])
            base_vel = np.random.uniform(-3, 3, 2) * anomaly_speed
            
            self.anomaly_gen.add_anomaly(
                anomaly_type=atype,
                size=size,
                velocity=tuple(base_vel),
                pulsing=(i % 3 == 0),
                trail=(i % 4 == 0),
                rotation_speed=np.random.uniform(-2, 2) if atype in ['drone', 'debris'] else 0
            )
    
    def add_vignette(self, frame: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Add vignette effect (darker corners)."""
        if strength <= 0:
            return frame
        
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Distance from center, normalized
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        
        # Vignette mask
        vignette = 1 - strength * (dist_norm ** 2)
        
        return frame * vignette.astype(np.float32)
    
    def add_noise(self, frame: np.ndarray, noise_level: float = 0.02, 
                  noise_type: str = 'gaussian') -> np.ndarray:
        """Add noise to frame."""
        if noise_level <= 0:
            return frame
            
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, frame.shape).astype(np.float32)
            noisy = frame + noise
        elif noise_type == 'salt_pepper':
            noisy = frame.copy()
            # Salt
            salt_mask = np.random.random(frame.shape) < noise_level / 2
            noisy[salt_mask] = 1.0
            # Pepper
            pepper_mask = np.random.random(frame.shape) < noise_level / 2
            noisy[pepper_mask] = 0.0
        elif noise_type == 'poisson':
            # Poisson noise (shot noise)
            vals = len(np.unique(frame))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(frame * vals * (1 - noise_level * 10)) / float(vals)
        else:
            noisy = frame
            
        return np.clip(noisy, 0, 1)
    
    def add_blur(self, frame: np.ndarray, blur_amount: float = 1.0,
                blur_type: str = 'gaussian') -> np.ndarray:
        """Add blur to frame."""
        if blur_amount <= 0:
            return frame
        
        if blur_type == 'gaussian':
            ksize = int(blur_amount * 5) | 1
            blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_amount)
        elif blur_type == 'motion':
            # Horizontal motion blur
            ksize = int(blur_amount * 5) | 1
            kernel = np.zeros((ksize, ksize))
            kernel[ksize // 2, :] = 1.0 / ksize
            blurred = cv2.filter2D(frame, -1, kernel)
        elif blur_type == 'radial':
            # Radial blur (zoom blur) - simplified
            ksize = int(blur_amount * 3) | 1
            blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_amount * 0.5)
        else:
            blurred = frame
            
        return blurred
    
    def add_compression_artifacts(self, frame: np.ndarray, quality: int = 50) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        if quality >= 100:
            return frame
        
        # Convert to uint8, compress, decompress
        frame_uint8 = (frame * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame_uint8, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        
        return decoded.astype(np.float32) / 255.0
    
    def adjust_brightness_contrast(self, frame: np.ndarray, 
                                   brightness: float = 0.0,
                                   contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast."""
        adjusted = (frame - 0.5) * contrast + 0.5 + brightness
        return np.clip(adjusted, 0, 1)
    
    def generate_frame(self, frame_idx: int, 
                      noise_level: float = 0.02,
                      noise_type: str = 'gaussian',
                      blur_amount: float = 0.5,
                      blur_type: str = 'gaussian',
                      vignette: float = 0.0,
                      compression_quality: int = 100,
                      brightness: float = 0.0,
                      contrast: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single frame with all effects."""
        
        M = self.camera.get_transform(frame_idx)
        
        offset_x = self.terrain.shape[1] // 3
        offset_y = self.terrain.shape[0] // 3
        
        M_adjusted = M.copy()
        M_adjusted[0, 2] += offset_x
        M_adjusted[1, 2] += offset_y
        
        background = cv2.warpAffine(
            self.terrain, M_adjusted, 
            (self.width, self.height),
            borderMode=cv2.BORDER_REFLECT
        )
        
        anomaly_layer, gt_mask = self.anomaly_gen.get_anomaly_mask(frame_idx)
        frame = np.where(anomaly_layer > 0.1, anomaly_layer, background)
        
        # Apply effects in order
        frame = self.adjust_brightness_contrast(frame, brightness, contrast)
        frame = self.add_blur(frame, blur_amount, blur_type)
        frame = self.add_vignette(frame, vignette)
        frame = self.add_noise(frame, noise_level, noise_type)
        
        if compression_quality < 100:
            frame = self.add_compression_artifacts(frame, compression_quality)
        
        return frame.astype(np.float32), gt_mask
    
    def generate_video(self,
                      output_path: str,
                      num_frames: int = 100,
                      noise_level: float = 0.02,
                      noise_type: str = 'gaussian',
                      blur_amount: float = 0.5,
                      blur_type: str = 'gaussian',
                      vignette: float = 0.0,
                      compression_quality: int = 100,
                      brightness: float = 0.0,
                      contrast: float = 1.0,
                      save_ground_truth: bool = True,
                      save_frames: bool = False,
                      save_config: bool = True) -> Dict[str, str]:
        """Generate complete synthetic video with all outputs."""
        
        print(f"\nGenerating {num_frames} frames...")
        print(f"  Noise: {noise_level} ({noise_type})")
        print(f"  Blur: {blur_amount} ({blur_type})")
        print(f"  Vignette: {vignette}")
        print(f"  Compression: {compression_quality}%")
        print(f"  Brightness: {brightness}, Contrast: {contrast}")
        
        outputs = {'video': output_path}
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height), isColor=False)
        
        # Ground truth writer
        gt_out = None
        if save_ground_truth:
            gt_path = output_path.replace('.mp4', '_groundtruth.mp4')
            gt_out = cv2.VideoWriter(gt_path, fourcc, self.fps,
                                    (self.width, self.height), isColor=False)
            outputs['ground_truth'] = gt_path
        
        # Frames directory
        frames_dir = None
        if save_frames:
            frames_dir = output_path.replace('.mp4', '_frames')
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(os.path.join(frames_dir, 'input'), exist_ok=True)
            os.makedirs(os.path.join(frames_dir, 'gt'), exist_ok=True)
            outputs['frames_dir'] = frames_dir
        
        # Generate frames
        for i in range(num_frames):
            frame, gt_mask = self.generate_frame(
                i, noise_level, noise_type, blur_amount, blur_type,
                vignette, compression_quality, brightness, contrast
            )
            
            frame_uint8 = (frame * 255).astype(np.uint8)
            out.write(frame_uint8)
            
            if gt_out is not None:
                gt_out.write(gt_mask)
            
            if frames_dir is not None:
                cv2.imwrite(os.path.join(frames_dir, 'input', f'frame_{i:05d}.png'), frame_uint8)
                cv2.imwrite(os.path.join(frames_dir, 'gt', f'frame_{i:05d}.png'), gt_mask)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{num_frames} frames")
        
        out.release()
        if gt_out is not None:
            gt_out.release()
            print(f"Ground truth saved to: {gt_path}")
        
        if frames_dir is not None:
            print(f"Individual frames saved to: {frames_dir}/")
        
        # Save config
        if save_config:
            config_path = output_path.replace('.mp4', '_config.json')
            full_config = {
                **self.config,
                'video_settings': {
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps,
                    'num_frames': num_frames,
                    'noise_level': noise_level,
                    'noise_type': noise_type,
                    'blur_amount': blur_amount,
                    'blur_type': blur_type,
                    'vignette': vignette,
                    'compression_quality': compression_quality,
                    'brightness': brightness,
                    'contrast': contrast
                }
            }
            # Convert numpy types for JSON
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            full_config = json.loads(json.dumps(full_config, default=convert))
            
            with open(config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
            outputs['config'] = config_path
            print(f"Config saved to: {config_path}")
        
        print(f"Video saved to: {output_path}")
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic airborne video for anomaly detection testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic video with defaults
  python generate_synthetic_video.py -o output.mp4
  
  # High quality urban scene with multiple anomalies
  python generate_synthetic_video.py -o urban.mp4 --terrain urban --anomalies 5 --motion sweep
  
  # Noisy, compressed footage simulating poor quality
  python generate_synthetic_video.py -o poor_quality.mp4 --noise 0.08 --noise-type salt_pepper --compression 30
  
  # Drone hovering with turbulence
  python generate_synthetic_video.py -o hover.mp4 --motion hover --turbulence 3.0 --vignette 0.4
  
  # Export individual frames for analysis
  python generate_synthetic_video.py -o analysis.mp4 --save-frames
        """
    )
    
    # Basic parameters
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
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Terrain parameters
    parser.add_argument('--terrain', type=str, default='mixed',
                       choices=['urban', 'rural', 'desert', 'forest', 'mixed', 'water', 'mountain'],
                       help='Terrain type')
    parser.add_argument('--texture-scale', type=float, default=1.0,
                       help='Terrain texture scale (0.5-2.0)')
    parser.add_argument('--terrain-contrast', type=float, default=1.0,
                       help='Terrain contrast (0.5-2.0)')
    
    # Camera motion parameters
    parser.add_argument('--motion', type=str, default='linear',
                       choices=['linear', 'sinusoidal', 'circular', 'hover', 'dive', 'sweep'],
                       help='Camera motion type')
    parser.add_argument('--velocity-x', type=float, default=2.0,
                       help='Camera X velocity (pixels/frame)')
    parser.add_argument('--velocity-y', type=float, default=1.0,
                       help='Camera Y velocity (pixels/frame)')
    parser.add_argument('--rotation-velocity', type=float, default=0.1,
                       help='Camera rotation velocity (degrees/frame)')
    parser.add_argument('--zoom-velocity', type=float, default=0.0,
                       help='Camera zoom velocity (per frame, + = zoom in)')
    parser.add_argument('--turbulence', type=float, default=0.0,
                       help='Camera shake/turbulence amount')
    
    # Anomaly parameters
    parser.add_argument('--anomalies', type=int, default=2,
                       help='Number of anomaly objects')
    parser.add_argument('--anomaly-types', type=str, nargs='+',
                       choices=['ufo', 'aircraft', 'bird', 'debris', 'drone', 'missile', 'balloon', 'helicopter'],
                       help='Specific anomaly types to use')
    parser.add_argument('--anomaly-speed', type=float, default=1.0,
                       help='Anomaly speed multiplier')
    parser.add_argument('--anomaly-size-min', type=int, default=15,
                       help='Minimum anomaly size (pixels)')
    parser.add_argument('--anomaly-size-max', type=int, default=40,
                       help='Maximum anomaly size (pixels)')
    
    # Degradation parameters
    parser.add_argument('--noise', type=float, default=0.02,
                       help='Noise level (0-1)')
    parser.add_argument('--noise-type', type=str, default='gaussian',
                       choices=['gaussian', 'salt_pepper', 'poisson'],
                       help='Type of noise to add')
    parser.add_argument('--blur', type=float, default=0.5,
                       help='Blur amount (0-5)')
    parser.add_argument('--blur-type', type=str, default='gaussian',
                       choices=['gaussian', 'motion', 'radial'],
                       help='Type of blur to add')
    parser.add_argument('--vignette', type=float, default=0.0,
                       help='Vignette strength (0-1)')
    parser.add_argument('--compression', type=int, default=100,
                       help='JPEG compression quality (1-100, 100=no compression)')
    parser.add_argument('--brightness', type=float, default=0.0,
                       help='Brightness adjustment (-0.5 to 0.5)')
    parser.add_argument('--contrast', type=float, default=1.0,
                       help='Contrast adjustment (0.5-2.0)')
    
    # Output options
    parser.add_argument('--no-groundtruth', action='store_true',
                       help='Skip saving ground truth masks')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save individual frames as PNG')
    parser.add_argument('--no-config', action='store_true',
                       help='Skip saving config JSON')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SYNTHETIC AIRBORNE VIDEO GENERATOR (Extended)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Output: {args.output}")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"  Frames: {args.num_frames}")
    print(f"\nScene:")
    print(f"  Terrain: {args.terrain} (scale={args.texture_scale}, contrast={args.terrain_contrast})")
    print(f"  Motion: {args.motion}")
    print(f"  Camera velocity: ({args.velocity_x}, {args.velocity_y})")
    print(f"  Rotation: {args.rotation_velocity} deg/frame")
    print(f"  Zoom: {args.zoom_velocity}/frame, Turbulence: {args.turbulence}")
    print(f"\nAnomalies:")
    print(f"  Count: {args.anomalies}")
    print(f"  Types: {args.anomaly_types or 'all'}")
    print(f"  Speed: {args.anomaly_speed}x")
    print(f"  Size: {args.anomaly_size_min}-{args.anomaly_size_max}px")
    print(f"\nDegradation:")
    print(f"  Noise: {args.noise} ({args.noise_type})")
    print(f"  Blur: {args.blur} ({args.blur_type})")
    print(f"  Vignette: {args.vignette}")
    print(f"  Compression: {args.compression}%")
    print(f"  Brightness: {args.brightness}, Contrast: {args.contrast}")
    
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
        motion_velocity=(args.velocity_x, args.velocity_y),
        rotation_velocity=args.rotation_velocity,
        zoom_velocity=args.zoom_velocity,
        turbulence=args.turbulence,
        num_anomalies=args.anomalies,
        anomaly_types=args.anomaly_types,
        anomaly_speed=args.anomaly_speed,
        anomaly_size_range=(args.anomaly_size_min, args.anomaly_size_max),
        texture_scale=args.texture_scale,
        terrain_contrast=args.terrain_contrast
    )
    
    # Generate video
    outputs = generator.generate_video(
        output_path=args.output,
        num_frames=args.num_frames,
        noise_level=args.noise,
        noise_type=args.noise_type,
        blur_amount=args.blur,
        blur_type=args.blur_type,
        vignette=args.vignette,
        compression_quality=args.compression,
        brightness=args.brightness,
        contrast=args.contrast,
        save_ground_truth=not args.no_groundtruth,
        save_frames=args.save_frames,
        save_config=not args.no_config
    )
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    print(f"\nTo run anomaly detection:")
    print(f"  1. Edit main.py: VIDEO_PATH = '{args.output}'")
    print(f"  2. Run: python3 main.py")


if __name__ == "__main__":
    main()
