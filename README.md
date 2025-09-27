# Low-Light Video Temporal Filtering (LLVTF)

A toolkit for video denoising and processing with temporal filters, supporting both low-light video enhancement and general video processing tasks.

## Overview

This project provides a configurable pipeline for denoising videos using temporal filters without AI. All processing methods are implemented as filters that can be chained into a pipeline. The toolkit also includes specialized utilities for handling NaN-filled regions in multi-dimensional arrays.

## Features

### Supported Filters

#### Temporal Denoising Filters
- **Wiener Filter**: Adaptive Wiener filter from scipy.signal with auto-noise estimation
- **Kalman Filter**: Temporal trajectory modeling with position, velocity, and acceleration states
- **Savitzky-Golay Filter**: Polynomial smoothing for temporal signals
- **DCT Thresholding**: Frequency-domain denoising with hard thresholding
- **Pattern Thresholding**: User-defined pattern removal with orthonormal basis

#### Calibration Filters
- **Dark Frame Subtraction**: Removes fixed pattern noise using dark reference frames
- **Flat Field Correction**: Corrects for uneven illumination using flat reference frames

## Project Structure

```
low-light-video-temporal-filtering/
├── llvtf.py              # Main video processing pipeline
├── filters.py            # All filter implementations
├── utils.py              # Video I/O and utility handling utilities
├── requirements.txt      # Python dependencies
├── cfgs/                 # Configuration files
│   ├── general.yaml      # General purpose configuration
│   ├── meteor.yaml       # Meteor video processing preset
│   └── default/         # Default filter configurations
└── video-samples/        # Sample videos for testing
```

## Installation

### Requirements

- Python 3.8+
- FFmpeg (for video encoding/decoding)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/low-light-video-temporal-filtering.git
cd low-light-video-temporal-filtering

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Video Denoising

```bash
# Process video with general configuration
python llvtf.py input_video.mp4 --config cfgs/general.yaml --output denoised_video.mp4

# Process meteor video with specialized configuration
python llvtf.py meteor_video.mp4 --config cfgs/meteor.yaml --output meteor_denoised.mp4
```

## Filter Parameters

### Wiener Filter
- `window_size`: Size of the filtering window (e.g., [19, 1, 1, 1])
- `noise`: Noise estimation method ("auto" or "pixel-wise")
- `var_scale`: Scaling factor for pixel-wise variance estimation

### Kalman Filter
- `padding_size`: Temporal padding size
- `block_size`: Spatial block size for processing
- `position_noise`: Process noise for position
- `velocity_noise`: Process noise for velocity
- `acceleration_noise`: Process noise for acceleration
- `observation_noise`: Measurement noise parameter

### DCT Thresholding Filter
- `kernel_size`: DCT kernel dimensions (e.g., [3, 3, 3])
- `threshold`: Hard threshold for frequency coefficients
- `down_scale`: Downscaling factors for efficiency

### Pattern Thresholding Filter
- `pattern`: User-defined pattern to remove
- `threshold`: Detection threshold for pattern removal
- `down_scale`: Downscaling factors for processing

### NaN Hypercube Handler
- `method`: Filling method ('interpolation', 'median', 'nearest')
- `kernel_size`: Median filter kernel size (for median method)
- `interp_method`: Interpolation method ('linear', 'nearest', 'cubic')

## Configuration Examples

### General Purpose Configuration (cfgs/general.yaml)
```yaml
pipe:
  - filter: wiener
    window_size: [19, 1, 1, 1]
  - filter: dct
    kernel_size: [3, 1, 1]
    threshold: 0.1

frames_per_clip: 36
```

### Meteor Video Configuration (cfgs/meteor.yaml)
```yaml
pipe:
  - filter: wiener
    window_size: [19, 1, 1, 1]
  - filter: dct
    kernel_size: [3, 3, 3]
    threshold: 0.07
    down_scale: [0.5, 0.33, 0.33]
  - filter: pattern
    pattern: [[[-1]], [[2]], [[-1]]]
    threshold: 0.02
    down_scale: [1.0, 0.33, 0.33]

frames_per_clip: 36
```

## API Reference

### Core Functions

#### `llvtf.py` - Main Pipeline
- `load_video_to_ndarrays()`: Load video into memory-efficient clips
- `temporal_stitch_frames()`: Stitch processed clips back together
- `save_ndarray_to_video()`: Save processed array to video file

#### `filters.py` - Filter Implementations
- `BaseFilter`: Abstract base class for all filters
- `WienerFilter`, `KalmanFilter`, `SavitzkyGolayFilter`: Temporal filters
- `DCTThresholdingFilter`, `PatternThresholdingFilter`: Frequency-domain filters
- `DarkFilter`, `FlatFilter`: Calibration filters

#### `utils.py` - Utilities
- Video I/O functions with FFmpeg integration
- Configuration loading with OmegaConf
- Memory management and garbage collection

## Advanced Usage

### Custom Filter Pipeline
```python
from filters import video_filters
from utils import load_video_to_ndarrays, temporal_stitch_frames

# Create custom filter chain
filters = [
    video_filters['wiener'](window_size=[15, 1, 1, 1]),
    video_filters['kalman'](block_size=16, position_noise=0.1),
    video_filters['dct'](kernel_size=[5, 5, 5], threshold=0.05)
]

# Process video
video_clips = load_video_to_ndarrays('input.mp4')
processed_clips = []
for clip in video_clips:
    for filter in filters:
        clip = filter(clip)
    processed_clips.append(clip)

# Stitch and save result
result = temporal_stitch_frames(processed_clips)
```

### Real-time Processing
The toolkit supports processing videos in chunks, making it suitable for real-time applications or memory-constrained environments.

## Performance Considerations

- **Memory Usage**: Videos are processed in clips to reduce memory footprint
- **GPU Acceleration**: PyTorch-based filters support GPU acceleration
- **Parallel Processing**: Filters can be optimized for parallel execution

## Known Issues and Limitations
- Compute/memory efficiency is not fully optimized for large videos
- Not well tested
- This README is generated by AI without proofreading
