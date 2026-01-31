# Airborne Visual Self-Consistency Anomaly Detection (AVSCA)

Self-Consistency Based Anomaly Detection for Airborne Imagery using classical computer vision techniques.

This pipeline detects anomalies in airborne imagery (from jets, UAVs, drones) using physical and geometric self-consistency analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Generating Synthetic Video](#generating-synthetic-video)
- [Running Anomaly Detection](#running-anomaly-detection)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Understanding Output Metrics](#understanding-output-metrics)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)

## Features

- **7-Stage Pipeline**: Comprehensive anomaly detection through multiple consistency checks
- **Physical Degradation Analysis**: Estimates blur, noise, and contrast
- **Motion-Based Detection**: Uses optical flow and RANSAC for background motion estimation
- **Multi-Scale Analysis**: Validates anomalies across scale space
- **Temporal Consistency**: Tracks anomalies over time to filter noise
- **Synthetic Video Generator**: Create test videos with controllable anomalies

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**

```bash
cd /home/pratosh/Desktop/Desktop/programming/code/avsca
```

2. **Create a virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import cv2; import numpy; import scipy; print('All dependencies installed!')"
```

## Quick Start

### Step 1: Generate a Synthetic Test Video

```bash
python generate_synthetic_video.py -o test_video.mp4 -n 100 --anomalies 3
```

This creates:
- `test_video.mp4` - Synthetic aerial footage with anomalies
- `test_video_groundtruth.mp4` - Ground truth anomaly masks

### Step 2: Run Anomaly Detection

Edit `main.py` to set the video path:

```python
VIDEO_PATH = "test_video.mp4"  # Your video file
```

Then run:

```bash
python main.py
```

Output: `anomaly_detection_output.mp4` with detected anomalies highlighted.

## Generating Synthetic Video

The synthetic video generator creates realistic aerial footage for testing.

### Basic Usage

```bash
python generate_synthetic_video.py -o output.mp4
```

### Full Options

```bash
python generate_synthetic_video.py \
    -o synthetic_aerial.mp4 \
    -n 100 \
    -W 640 \
    -H 480 \
    --fps 30 \
    --terrain mixed \
    --motion linear \
    --anomalies 3 \
    --noise 0.02 \
    --blur 0.5 \
    --seed 42
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-o, --output` | `synthetic_aerial.mp4` | Output video path |
| `-n, --num-frames` | `100` | Number of frames |
| `-W, --width` | `640` | Video width |
| `-H, --height` | `480` | Video height |
| `--fps` | `30.0` | Frames per second |
| `--terrain` | `mixed` | Terrain type: `urban`, `rural`, `desert`, `forest`, `mixed` |
| `--motion` | `linear` | Camera motion: `linear`, `sinusoidal`, `circular` |
| `--anomalies` | `2` | Number of anomaly objects |
| `--noise` | `0.02` | Gaussian noise level (0-1) |
| `--blur` | `0.5` | Blur amount (0-5) |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-groundtruth` | `False` | Skip saving ground truth masks |

### Terrain Types

- **urban**: Grid-like patterns simulating buildings/roads
- **rural**: Field-like striped patterns
- **desert**: Uniform sandy appearance
- **forest**: Dark, varied vegetation
- **mixed**: Combination of textures

### Motion Types

- **linear**: Straight-line camera movement
- **sinusoidal**: Oscillating motion (simulates turbulence)
- **circular**: Orbiting camera movement

### Anomaly Types

The generator creates various anomaly types:
- **UFO**: Elliptical bright objects
- **Aircraft**: Cross-shaped moving objects
- **Bird**: Small moving dots
- **Debris**: Irregular shaped objects

## Running Anomaly Detection

### Using the Main Script

```bash
python main.py
```

### Programmatic Usage

```python
from src.pipeline import run_anomaly_detection_pipeline

results = run_anomaly_detection_pipeline(
    video_path="your_video.mp4",
    max_frames=100,
    output_path="output.mp4"
)

# Access individual stage outputs
frames = results['frames']
motion_results = results['motion_results']
final_results = results['final_results']
```

### Using Individual Components

```python
from src.video_loader import VideoLoader
from src.image_degradation import ImageDegradationEstimator
from src.background_motion import BackgroundMotionEstimator

# Load video
loader = VideoLoader("video.mp4", max_frames=50)
frames = loader.load()

# Analyze degradation
estimator = ImageDegradationEstimator()
metrics = estimator.process_sequence(frames)

# Compute motion
motion_est = BackgroundMotionEstimator(model_type='affine')
result = motion_est.process_frame_pair(frames[0], frames[1])
```

## Project Structure

```
avsca/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── video_loader.py          # Video loading and preprocessing
│   ├── image_degradation.py     # Stage 1: Degradation estimation
│   ├── background_motion.py     # Stage 2: Motion estimation
│   ├── motion_consistency.py    # Stage 3: Motion consistency
│   ├── scale_space.py           # Stage 4: Scale-space analysis
│   ├── temporal_consistency.py  # Stage 5: Temporal consistency
│   ├── degradation_checker.py   # Stage 6: Degradation checking
│   ├── anomaly_fusion.py        # Stage 7: Score fusion
│   └── pipeline.py              # Main pipeline orchestration
├── main.py                      # Entry point
├── generate_synthetic_video.py  # Synthetic video generator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Pipeline Stages

### Stage 1: Image Degradation Estimation
Estimates physical degradation metrics (blur, noise, contrast) for context.

### Stage 2: Background Motion Estimation
Uses Farnebäck optical flow + RANSAC to estimate global camera motion.

### Stage 3: Motion Consistency Check
Identifies regions with residual motion that violates the static background assumption.

### Stage 4: Scale-Space Consistency
Validates anomalies across multiple pyramid scales (true features persist).

### Stage 5: Temporal Consistency
Tracks anomaly persistence over frames to filter transient noise.

### Stage 6: Degradation Consistency
Verifies anomaly regions degrade similarly to background (catches digital artifacts).

### Stage 7: Fusion
Combines all consistency scores into final anomaly map with confidence.

## Understanding Output Metrics

The pipeline outputs three key metrics for defense and surveillance applications:

### Anomaly Score (per-pixel, range 0-1)

A weighted fusion of 4 consistency checks that indicates how "anomalous" each pixel is:

```
anomaly_score = 0.30 × motion_inconsistency 
              + 0.25 × scale_inconsistency 
              + 0.30 × temporal_inconsistency
              × degradation_consistency^0.15
```

| Component | Weight | What it measures |
|-----------|--------|------------------|
| **Motion** | 30% | Does this region move differently from global camera motion? |
| **Scale** | 25% | Does the anomaly persist across image scales (not just noise)? |
| **Temporal** | 30% | Does it persist across multiple frames (not transient)? |
| **Degradation** | 15% | Does the region blur/noise consistently with the background? |

**Higher score = more likely a real anomaly** (foreign object, intruder, aircraft, drone, etc.)

### Confidence (per-pixel, range 0-1)

Measures how much the different detection methods **agree** with each other:

```
agreement = 1.0 - std(motion_score, scale_score, temporal_score)
confidence = agreement × anomaly_score
```

- **High confidence**: All 4 methods flag the same region → likely real threat
- **Low confidence**: Methods disagree → could be noise, weather, or sensor artifact

### Summary Statistics

| Metric | Calculation | Meaning |
|--------|-------------|---------|
| **Average Anomaly Score** | Mean of all pixel scores across all frames | Overall "suspiciousness" of the video |
| **Maximum Anomaly Score** | Highest single pixel score in any frame | The most anomalous point detected (worst case) |
| **Average Confidence** | Mean confidence across all frames | How reliable the detections are overall |

### Defense Relevance & Operational Thresholds

These metrics are designed for airborne surveillance and threat detection:

| Condition | Alert Level | Action |
|-----------|-------------|--------|
| `max > 0.8` AND `confidence > 0.5` | **HIGH ALERT** | Likely real object — immediate investigation |
| `max > 0.5` AND `confidence > 0.3` | **MEDIUM ALERT** | Possible threat — review footage |
| `avg > 0.2` | **ELEVATED** | Unusual activity detected — monitor |
| `avg < 0.1` AND `max < 0.3` | **ALL CLEAR** | Normal operations — no significant anomalies |

### Example Interpretation

```
Average anomaly score: 0.0643
Maximum anomaly score: 1.0000  
Average confidence: 0.0429
```

**Analysis:**
- **Low average (0.06)**: Most of the frame is normal background
- **Max = 1.0**: Something triggered maximum detection (could be a moving object)
- **Low confidence (0.04)**: Detections are scattered/inconsistent, suggesting natural motion (waves, trees) rather than a coherent threat like an aircraft or drone

For defense applications, you would typically alert on `max > 0.7 AND confidence > 0.4` to balance sensitivity with false positive rate.

## Configuration

### Pipeline Parameters

Edit parameters in `main.py` or when calling `run_anomaly_detection_pipeline()`:

```python
results = run_anomaly_detection_pipeline(
    video_path="video.mp4",
    max_frames=100,      # Max frames to process
    output_path="out.mp4"
)
```

## Contributing

Contributions welcome! Please ensure code follows the existing style and includes appropriate documentation.
