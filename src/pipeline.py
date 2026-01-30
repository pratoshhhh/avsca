"""
Main Pipeline

Run the complete anomaly detection pipeline.
"""

import numpy as np
import cv2

from .video_loader import VideoLoader
from .image_degradation import ImageDegradationEstimator
from .background_motion import BackgroundMotionEstimator
from .motion_consistency import MotionConsistencyChecker
from .scale_space import ScaleSpaceConsistencyAnalyzer
from .temporal_consistency import TemporalConsistencyAnalyzer
from .degradation_checker import PhysicalDegradationChecker
from .anomaly_fusion import AnomalyFusionEngine


def run_anomaly_detection_pipeline(video_path: str, 
                                   max_frames: int = 100,
                                   output_path: str = 'anomaly_output.mp4'):
    """
    Run the complete anomaly detection pipeline.
    
    Args:
        video_path: Path to input video
        max_frames: Maximum number of frames to process
        output_path: Path for output video
    """
    
    print("="*80)
    print("AIRBORNE ANOMALY DETECTION PIPELINE")
    print("="*80)
    
    # Load video
    print("\n[1/7] Loading video...")
    loader = VideoLoader(video_path, max_frames=max_frames, target_width=640)
    frames = loader.load()
    
    # Stage 1: Image degradation estimation
    print("\n[2/7] Stage 1: Estimating image degradation...")
    degradation_estimator = ImageDegradationEstimator()
    degradation_metrics = degradation_estimator.process_sequence(frames)
    
    # Stage 2: Background motion estimation
    print("\n[3/7] Stage 2: Estimating background motion...")
    motion_estimator = BackgroundMotionEstimator(model_type='affine')
    motion_results = []
    for i in range(min(len(frames) - 1, max_frames)):
        result = motion_estimator.process_frame_pair(frames[i], frames[i+1])
        motion_results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1} frame pairs")
    
    # Stage 3: Motion consistency check
    print("\n[4/7] Stage 3: Checking motion consistency...")
    consistency_checker = MotionConsistencyChecker(threshold_sigma=3.0, min_area=100)
    consistency_results = []
    for motion_result in motion_results:
        consistency = consistency_checker.compute_motion_inconsistency(
            motion_result['residual_magnitude']
        )
        consistency_results.append(consistency)
    
    # Stage 4: Scale-space analysis
    print("\n[5/7] Stage 4: Analyzing scale-space consistency...")
    scale_analyzer = ScaleSpaceConsistencyAnalyzer(num_scales=4)
    scale_results = []
    for consistency_result in consistency_results:
        scale_result = scale_analyzer.analyze_scale_consistency(
            consistency_result['inconsistency_map']
        )
        scale_results.append(scale_result)
    
    # Stage 5: Temporal consistency
    print("\n[6/7] Stage 5: Analyzing temporal consistency...")
    temporal_analyzer = TemporalConsistencyAnalyzer(window_size=5, min_persistence=0.4)
    temporal_results = []
    for i in range(len(consistency_results)):
        if i < len(scale_results):
            inconsistency = scale_results[i]['consistency_score']
        else:
            inconsistency = consistency_results[i]['inconsistency_map']
        
        temporal_result = temporal_analyzer.update_and_analyze(
            inconsistency,
            consistency_results[i]['binary_mask']
        )
        temporal_results.append(temporal_result)
    
    # Stage 6: Degradation consistency
    print("\n[7/7] Stage 6: Checking degradation consistency...")
    degradation_checker = PhysicalDegradationChecker(tolerance_sigma=2.0)
    degradation_check_results = []
    for i in range(min(len(frames), len(temporal_results))):
        global_metrics = {
            'blur': degradation_metrics['blur'][i],
            'noise': degradation_metrics['noise'][i],
            'contrast': degradation_metrics['contrast'][i]
        }
        check_result = degradation_checker.check_degradation_consistency(
            frames[i],
            temporal_results[i]['persistence_mask'],
            global_metrics
        )
        degradation_check_results.append(check_result)
    
    # Stage 7: Fusion
    print("\n[8/8] Stage 7: Fusing results...")
    fusion_engine = AnomalyFusionEngine(
        motion_weight=0.3,
        scale_weight=0.25,
        temporal_weight=0.3,
        degradation_weight=0.15
    )
    
    final_results = []
    for i in range(min(len(consistency_results), 
                       len(scale_results), 
                       len(temporal_results),
                       len(degradation_check_results))):
        
        fusion_result = fusion_engine.fuse_scores(
            motion_score=consistency_results[i]['inconsistency_map'],
            scale_score=scale_results[i]['consistency_score'],
            temporal_score=temporal_results[i]['temporal_score'],
            degradation_weight=degradation_check_results[i]['consistency_weight']
        )
        
        overlay = fusion_engine.generate_visualization(
            frames[i],
            fusion_result['anomaly_score'],
            fusion_result['confidence'],
            threshold=0.5
        )
        
        fusion_result['overlay'] = overlay
        final_results.append(fusion_result)
    
    # Export video
    print("\nExporting output video...")
    if len(final_results) > 0:
        h, w = final_results[0]['overlay'].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, loader.fps, (w, h))
        
        for result in final_results:
            frame_bgr = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Output saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"\nProcessed {len(frames)} frames")
    print(f"Detected {sum([r['num_regions'] for r in consistency_results])} total anomaly regions")
    
    avg_anomaly = np.mean([r['anomaly_score'].mean() for r in final_results])
    max_anomaly = np.max([r['anomaly_score'].max() for r in final_results])
    avg_confidence = np.mean([r['confidence'].mean() for r in final_results])
    
    print(f"\nAnomaly Statistics:")
    print(f"  Average anomaly score: {avg_anomaly:.4f}")
    print(f"  Maximum anomaly score: {max_anomaly:.4f}")
    print(f"  Average confidence: {avg_confidence:.4f}")
    
    return {
        'frames': frames,
        'degradation_metrics': degradation_metrics,
        'motion_results': motion_results,
        'consistency_results': consistency_results,
        'scale_results': scale_results,
        'temporal_results': temporal_results,
        'degradation_check_results': degradation_check_results,
        'final_results': final_results
    }
