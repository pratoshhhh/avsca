"""
Self-Consistency Based Anomaly Detection for Airborne Imagery

This package implements a classical computer vision pipeline for detecting
anomalies in airborne imagery (from jets, UAVs, drones) using physical
and geometric self-consistency analysis.

Author: Defense-Oriented Computer Vision Lab
Date: January 2026
"""

from .video_loader import VideoLoader
from .image_degradation import ImageDegradationEstimator
from .background_motion import BackgroundMotionEstimator
from .motion_consistency import MotionConsistencyChecker
from .scale_space import ScaleSpaceConsistencyAnalyzer
from .temporal_consistency import TemporalConsistencyAnalyzer
from .degradation_checker import PhysicalDegradationChecker
from .anomaly_fusion import AnomalyFusionEngine
from .pipeline import run_anomaly_detection_pipeline

__all__ = [
    'VideoLoader',
    'ImageDegradationEstimator',
    'BackgroundMotionEstimator',
    'MotionConsistencyChecker',
    'ScaleSpaceConsistencyAnalyzer',
    'TemporalConsistencyAnalyzer',
    'PhysicalDegradationChecker',
    'AnomalyFusionEngine',
    'run_anomaly_detection_pipeline',
]
