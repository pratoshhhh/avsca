"""
Data Loading Module

Loads and preprocesses video frames for analysis.
"""

import numpy as np
import cv2
from typing import List


class VideoLoader:
    """
    Loads and preprocesses video frames for analysis.
    Handles downsampling for computational efficiency.
    """
    
    def __init__(self, video_path: str, max_frames: int = 100, 
                 target_width: int = 640, grayscale: bool = True):
        self.video_path = video_path
        self.max_frames = max_frames
        self.target_width = target_width
        self.grayscale = grayscale
        self.frames = []
        self.fps = 0
        self.original_size = None
        
    def load(self) -> List[np.ndarray]:
        """Load video frames into memory"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly if video is too long
        frame_indices = np.linspace(0, total_frames - 1, 
                                   min(self.max_frames, total_frames), 
                                   dtype=int)
        
        print(f"Loading {len(frame_indices)} frames from {total_frames} total frames...")
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Store original size from first frame
            if self.original_size is None:
                self.original_size = (frame.shape[1], frame.shape[0])
            
            # Resize to target width maintaining aspect ratio
            aspect_ratio = frame.shape[0] / frame.shape[1]
            target_height = int(self.target_width * aspect_ratio)
            frame = cv2.resize(frame, (self.target_width, target_height))
            
            # Convert to grayscale if requested
            if self.grayscale:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            self.frames.append(frame)
        
        cap.release()
        print(f"Loaded {len(self.frames)} frames at {self.frames[0].shape} resolution")
        print(f"FPS: {self.fps:.2f}")
        
        return self.frames
