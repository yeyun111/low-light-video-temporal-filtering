import os
import subprocess
import base64
from typing import List

import logging

import numpy
import scipy
import av
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def load_video_to_ndarrays(video_path: str, frames_per_clip: int = 120, overlap_frames: int = 10) -> List[numpy.ndarray]:
    """
    Load video to ndarrays.

    Args:
        video_path (str): Video path. 8-bit.
        max_frames_per_clip (int, optional): Max frames per clip. Defaults to 120.
        overlap_frames (int, optional): Overlap frames. Defaults to 10.

    Returns:
        List[numpy.ndarray]: Ndarrays in float32.
    """
    clip_arrays = []
    with av.open(video_path, mode='r') as container:
        video_stream = container.streams.video[0]
        frames = []
        for frame in container.decode(video_stream):
            frames.append(frame.to_ndarray(format='rgb24'))
        frames_array = numpy.array(frames, dtype=numpy.float32) / 255.0

    clip_index = 0
    t0 = clip_index * (frames_per_clip - overlap_frames)
    while t0 + frames_per_clip <= len(frames_array):
        clip_arrays.append(frames_array[t0:t0 + frames_per_clip])
        clip_index += 1
        t0 = clip_index * (frames_per_clip - overlap_frames)
    if t0 < len(frames_array) - 1:
        clip_arrays.append(frames_array[t0:])
    
    return clip_arrays

def load_video_to_ndarray(video_path: str) -> numpy.ndarray:
    """
    Load video to ndarray. 8-bit.

    Args:
        video_path (str): Video path.

    Returns:
        numpy.ndarray: Ndarray in float32.
    """
    return load_video_to_ndarrays(video_path, frames_per_clip=99999999)[0]

def save_ndarray_to_video(frames_array: numpy.ndarray, output_video_path: str, ref_video_path: str = None, crf: int = 24, ffmpeg: str = 'ffmpeg') -> None:
    """
    Save ndarray to video. 8-bit.

    Args:
        frames_array (numpy.ndarray): Ndarray in float32. 8-bit.
        output_video_path (str): Output video path.
        ref_video_path (str, optional): Ref video path. Defaults to None.
        crf (int, optional): CRF. Defaults to 24.
    """

    if len(frames_array.shape) != 4 or frames_array.shape[3] != 3:
        raise ValueError('frames_array must be 4-dim with shape (T, H, W, 3)')

    frames_array = numpy.clip((frames_array * 255.0).round(), 0, 255).astype(numpy.uint8)

    random_suffix = base64.urlsafe_b64encode(os.urandom(4)).decode('utf-8').rstrip('=')
    _temp_video_path = f"_LLVTF_{random_suffix}.mp4"

    if ref_video_path is None or not os.path.exists(ref_video_path):
        has_audio = False
        frame_rate = 30
        width = frames_array.shape[2]
        height = frames_array.shape[1]
        logger.info(f"Warning: No reference video. Use default frame rate {frame_rate}, width {width}, height {height}.")
    else:
        with av.open(ref_video_path, mode='r') as ref_video_container:
            has_audio = len(ref_video_container.streams.audio) > 0
            frame_rate = ref_video_container.streams.video[0].average_rate
            width = ref_video_container.streams.video[0].width
            height = ref_video_container.streams.video[0].height

    with av.open(_temp_video_path, mode='w') as temp_video_container:
        video_stream = temp_video_container.add_stream('libx264', rate=frame_rate)
        assert frames_array.shape[1] == height and frames_array.shape[2] == width, \
            f"Frame dimensions {frames_array.shape[2]}x{frames_array.shape[1]} do not match ref {width}x{height}."
        video_stream.width = width
        video_stream.height = height
        video_stream.framerate = frame_rate
        video_stream.pix_fmt = 'yuv420p'
        video_stream.options = {'qp': '0'}
        for frame in frames_array:
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            frame.pts = None
            for packet in video_stream.encode(frame):
                temp_video_container.mux(packet)

        # Flush
        for packet in video_stream.encode(None):
            temp_video_container.mux(packet)
    
    # FFMPEG CLI Command to do the final encoding
    cmd = [
        ffmpeg,
        '-i', _temp_video_path,
    ]
    if has_audio:
        cmd.extend(["-i", ref_video_path])
    
    cmd.extend([
        "-map", "0:v", 
        "-c:v", "libx264",
        "-preset", "veryslow", 
        "-crf", f"{crf}",
        "-pix_fmt", "yuv420p",
        "-vsync", "0",
    ])
    if has_audio:
        cmd.extend([
            "-map", "1:a",
            "-c:a", "copy",
        ])
    cmd.extend([
        "-y", 
        output_video_path
    ])

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.info(f"FFmpeg error: {result.stderr.decode()}")
    
    os.remove(_temp_video_path)


def temporal_stitch_frames(frames_array_list: List[numpy.ndarray], overlap_frames: int = 10) -> numpy.ndarray:
    """
    Temporal stitch frames.

    Args:
        frames_array_list (List[numpy.ndarray]): List of ndarrays in float32.
        overlap_frames (int, optional): Overlap frames. Defaults to 10.

    Returns:
        numpy.ndarray: Stitched ndarray in float32.
    """
    if len(frames_array_list) == 0:
        raise ValueError('frames_array_list is empty.')

    if len(frames_array_list) == 1:
        return frames_array_list[0]
    
    # Use clip 0 as the ref clip
    t, h, w, c = frames_array_list[0].shape
    num_clips = len(frames_array_list)

    dtype = frames_array_list[0].dtype
    T = sum([frames_array.shape[0] for frames_array in frames_array_list]) - (num_clips - 1) * overlap_frames
    trans_weight = numpy.linspace(0, 1, overlap_frames + 2, dtype=dtype)[1:-1]
    
    l_mask = numpy.ones(t, dtype=dtype)
    l_mask[:overlap_frames] = trans_weight
    r_mask = numpy.ones(t, dtype=dtype)
    r_mask[-overlap_frames:] = trans_weight[::-1]
    l_mask = l_mask.reshape(-1, 1, 1, 1)
    r_mask = r_mask.reshape(-1, 1, 1, 1)
    m_mask = l_mask * r_mask

    stitched_frames_array = numpy.zeros((T, h, w, c), dtype=dtype)
    stitched_frames_array[:t] = frames_array_list[0] * r_mask
    for i in range(1, num_clips):
        t0 = i * (t - overlap_frames)
        if i < num_clips - 1:
            stitched_frames_array[t0:t0+t] += frames_array_list[i] * m_mask
        else:
            t = frames_array_list[i].shape[0]
            stitched_frames_array[t0:] += frames_array_list[i] * l_mask[:t]

    return stitched_frames_array

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str, default='', help="Output video path. Defaults to input path.")
    parser.add_argument("--crf", type=int, default=24, help="CRF. Defaults to 24.")
    parser.add_argument("--frames_per_clip", type=int, default=60, help="Max frames per clip. Defaults to 120.")
    parser.add_argument("--overlap_frames", type=int, default=10, help="Overlap frames. Defaults to 10.")
    parser.add_argument("--filter", type=str, default='wiener', help="Filter. Defaults to wiener.")
    parser.add_argument("--filter_size", type=int, default=11, help="Filter size")
    parser.add_argument("--ffmpeg", type=str, default='ffmpeg', help="FFmpeg binary path. Defaults to ffmpeg.")
    args = parser.parse_args()
    
    print("\n#####################################################################")
    print("########### Low Light Video Temporal Filtering without AI ###########")
    print("#####################################################################")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("####################################################################")

    if args.output == '':
        basename, ext = os.path.splitext(args.input)
        args.output = basename + '_denoised' + ext

    clip_array_list = load_video_to_ndarrays(args.input, args.frames_per_clip, args.overlap_frames)
    num_clips = len(clip_array_list)
    logger.info(f"Loaded {args.input} to {num_clips} clips.")

    # Apply filter to denoise
    denoised_array_list = []
    H, W = clip_array_list[0].shape[1:3]
    for i, clip_array in enumerate(clip_array_list):
        logger.info(f"Denoising Clip: {i + 1}/{num_clips}, Shape: {clip_array.shape}, with {args.filter} filter ...")
        t_half = clip_array.shape[0] // 2
        filter_size = min(t_half * 2 + 1, args.filter_size)
        padding_size = filter_size // 4 * 2 + 1
        clip_array = numpy.pad(clip_array, ((padding_size, padding_size), (0, 0), (0, 0), (0, 0)), 'reflect')

        if args.filter == 'wiener':
            clip_array = scipy.signal.wiener(clip_array, (filter_size, 1, 1, 1))
        
        elif args.filter == 'wiener_pixwise':

            logger.info("Extracting noise variance map ...")
            # Extract pix-wise variance map with medfilt
            var_maps = []
            for t0 in range(0, clip_array.shape[0] - filter_size + 1):
                t1 = t0 + filter_size
                clip_for_var = clip_array[t0:t1]
                var_map = clip_for_var.var(axis=(0, -1))
                var_maps.append(var_map)

            var_maps = numpy.stack(var_maps)
            var_map = numpy.median(var_maps, axis=0)

            logger.info("Pix-wise denoising with wiener filter ...")
            # pix-wise denoise with wiener filter
            denoised = numpy.zeros_like(clip_array)
            for h in tqdm(range(H)):
                for w in range(W):
                    noise_var = var_map[h, w] * 1.414 # empirical factor median --> mean
                    if noise_var > 0:
                        clip_array[:, h, w, :] = scipy.signal.wiener(clip_array[:, h, w, :], (filter_size, 1), noise_var)
                    else:
                        clip_array[:, h, w, :] = clip_array[:, h, w, :]
        
        elif args.filter == 'median':
            clip_array = scipy.signal.medfilt(clip_array, (filter_size, 1, 1, 1))
        
        else:
            raise ValueError(f"Unknown filter: {args.filter}")
        
        clip_array = clip_array[padding_size:-padding_size]
        denoised_array_list.append(clip_array)

    logger.info(f"Exporting denoised video to {args.output} ...")
    denoised_frames_array = temporal_stitch_frames(denoised_array_list, args.overlap_frames)
    save_ndarray_to_video(denoised_frames_array, args.output, args.input, crf=args.crf, ffmpeg=args.ffmpeg)
    logger.info(f"Done!")
