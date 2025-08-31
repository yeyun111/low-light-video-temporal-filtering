import os

from scipy.ndimage import gaussian_filter
from omegaconf import OmegaConf

from filters import video_filters
from utils import load_video_to_ndarrays, temporal_stitch_frames, save_ndarray_to_video, \
    logger, load_config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--config", type=str, default="cfgs/meteor.yaml", help="config file")
    parser.add_argument("--output", type=str, default='', help="Output video path")
    args = parser.parse_args()
    if args.output == '':
        basename, ext = os.path.splitext(args.input)
        args.output = basename + '_denoised' + ext

    cfg = load_config(args.config)

    print("\n#####################################################################")
    print("########### Low Light Video Temporal Filtering without AI ###########")
    print("#####################################################################")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    for k, v in cfg.items():
        if k == 'pipe':
            print("filters pipeline:")
            for vf_cfg in v:
                filter_id = vf_cfg["filter"]
                print(f"  {filter_id}")
                for pname, pval in vf_cfg.items():
                    if pname != "filter":
                        print(f"    {pname}: {pval}")
        else:
            print(f"{k}: {v}")
    print("####################################################################")

    # Load video to array list
    video_array_list = load_video_to_ndarrays(args.input, cfg.frames_per_clip, cfg.overlap_frames)
    num_videos = len(video_array_list)
    
    logger.info(
        f"Loaded {args.input} to {num_videos} clips with "
        f"{','.join([str(video_array.shape[0]) for video_array in video_array_list])} frames."
    )

    # Initialize filters for the pipeline
    pipe_filters = []
    for vf_cfg in cfg.pipe:
        filter_id = vf_cfg.pop("filter")
        pipe_filters.append(video_filters[filter_id](**OmegaConf.to_object(vf_cfg)))
    
    # Apply filter to denoise
    denoised_array_list = []
    H, W = video_array_list[0].shape[1:3]
    for i, video_array in enumerate(video_array_list):
        for vf in pipe_filters:
            logger.info(f"Denoising Video Clip: {i + 1}/{num_videos}, Shape: {video_array.shape}, with {vf.id} filter ...")
            video_array = vf(video_array)
        denoised_array_list.append(video_array)

    logger.info(f"Exporting denoised video to {args.output} ...")
    denoised_frames_array = temporal_stitch_frames(denoised_array_list, cfg.overlap_frames)
    save_ndarray_to_video(denoised_frames_array, args.output, args.input, crf=cfg.output_crf, ffmpeg=cfg.ffmpeg)
    logger.info(f"Done!")
