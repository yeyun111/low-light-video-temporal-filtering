# Low-Light Video Temporal Filtering (LLVTF)

A non-AI approach to denoise with temporal filters for low-light videos.

## Overview

This project provides a configurable pipeline for denoising low-light videos. All the processing methods are implemented as filters that can be chained into a pipeline.   

The following filters are supported:

- **Wiener Filter**: Wiener filter from scipy.signal.
- **Kalman Filter**: Kalman filter from simdkalman. It models the temporal trajectory of each pixel as a dynamic system with position, velocity, and optional acceleration states.
- **Dark Frame Subtraction**: Removes fixed pattern noise using a dark reference frame.
- **Flat Field Correction**: Corrects for uneven illumination using a flat reference frame.

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

Basic usage:

```bash
python llvtf.py input_video.mp4 --config cfgs/general.yaml --output denoised_video.mp4
```

## Filter Parameters

### Wiener Filter

- `window_size`: Size of the filtering window
- `noise`: Noise estimation method ("auto" or "pixel-wise")
- `var_scale`: Scaling factor for pixel-wise variance estimation

#### Kalman Filter

- `padding_size`: Temporal padding size
- `block_size`: Spatial block size for processing
- `position_noise`: Process noise for position
- `velocity_noise`: Process noise for velocity
- `acceleration_noise`: Process noise for acceleration (set to >0 to enable acceleration modeling)
- `observation_noise`: Measurement noise parameter


#### Dark Frame Subtraction

- `video_path`: Path to the dark reference video (lens cap on)

#### Flat Field Correction

- `video_path`: Path to the flat field reference video (uniform light source)
- `smooth_size`: Smoothing parameter for the flat field

## Known Issues

- Compute/memory efficiency is not optimized at all.
- No full tests are done.
