# Temporal Filtering without AI for Low Light Video Denoising

## Requirements 
```shell
pip install -r requirements.txt
```

## Example
```shell
python llvtf.py input.mp4 --output output.mp4 --filter wiener --filter_size 11
```

## Parameters
| Parameter | Description | Default |
| -------- | -------- | -------- |
| input | Input video path |  |
| output | Output video path | [input]_denoised.[ext] |
| filter | Filter type | wiener, median, wiener_pixwise |
| filter_size | Filter size | 11 |
| frames_per_clip | Frames per clip | 60 |
| overlap_frames | Overlap frames | 10 |
| ffmpeg | FFmpeg binary path | ffmpeg |
| crf | CRF for encoding output video | 24 |

## Notes
- wiener_pixwise is designed for spatially adaptive noise reduction for better uniformity and is experimental.
