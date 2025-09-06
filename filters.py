import os
import sys
import inspect
from abc import ABC, abstractmethod

import numpy
from scipy.signal import savgol_filter, wiener
from scipy.ndimage import gaussian_filter
import torch
from omegaconf import OmegaConf
import simdkalman
from tqdm import tqdm

from utils import load_video_to_ndarrays, extract_average_frame, logger


class BaseFilter(ABC):
    """
    Base class for all filters. Handel padding and de-padding only.
    """
    id = "base"
    def __init__(self, **kwargs):
        base_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "cfgs/default/base.yaml"))
        for key, value in base_cfg.items():
            setattr(self, key, value)
        default_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), f"cfgs/default/{self.id}.yaml"))
        for key, value in default_cfg.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        pass

    def __call__(self, video_array: numpy.ndarray) -> numpy.ndarray:
        if self.padding_size is None:
            return self.filter(video_array)
        else:
            padded = numpy.pad(
                video_array, 
                [(p, p) for p in self.padding_size], 
                mode=self.padding_mode
            )
            tp, hp, wp, cp = padded.shape
            filtered = self.filter(padded)
            depadded = filtered[self.padding_size[0]:tp-self.padding_size[0], 
                                self.padding_size[1]:hp-self.padding_size[1], 
                                self.padding_size[2]:wp-self.padding_size[2], 
                                self.padding_size[3]:cp-self.padding_size[3]]
            return depadded


class SavitzkyGolayFilter(BaseFilter):
    """
    Savitzky Golay filter for temporal denoising using scipy.signal.savgol_filter
    """
    id = "savgol"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding_mode = None
    
    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        return savgol_filter(video_array, self.window_size, self.polyorder, self.deriv, self.delta, mode="mirror")


class WienerFilter(BaseFilter):
    """
    Wiener filter for temporal denoising using scipy.signal.wiener
    """
    id: str = "wiener"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.padding_size is None:
            self.padding_size = (
                self.window_size[0] // 2, 
                self.window_size[1] // 2, 
                self.window_size[2] // 2, 
                0
            )
    
    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        if self.noise == "auto":
            return wiener(video_array, self.window_size)
        else:
            T, H, W, C = video_array.shape
            t_window_size = self.window_size[0]
            var_maps = []
            for t0 in range(0, video_array.shape[0] - t_window_size + 1):
                t1 = t0 + t_window_size
                clip_for_var = video_array[t0:t1]
                var_map = clip_for_var.var(axis=(0, -1))
                var_maps.append(var_map)

            var_maps = numpy.stack(var_maps)
            var_map = numpy.median(var_maps, axis=0)

            # pix-wise denoise with wiener filter
            denoised = numpy.zeros_like(video_array)
            for h in tqdm(range(H)):
                for w in range(W):
                    noise_var = var_map[h, w] * self.var_scale # empirical factor median --> mean
                    if noise_var > 0:
                        denoised[:, h, w, :] = wiener(video_array[:, h, w, :], (t_window_size, 1), noise_var)
                    else:
                        denoised[:, h, w, :] = video_array[:, h, w, :]
            return denoised


class KalmanFilter(BaseFilter):
    """
    Kalman filter for temporal denoising using simdkalman, following the official example for 1d motion.
    The acceleration modeling is added and is optional.
    """
    id: str = "kalman"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if numpy.any(self.padding_size[1:]) != 0:
            raise ValueError("Kalman filter support temporal only. padding_size must be (x, 0, 0, 0)")
        self.model_acceleration = self.acceleration_noise > 0
        if self.model_acceleration:
            M_transition = numpy.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])
            M_process_noise = numpy.diag([self.position_noise, self.position_noise, self.acceleration_noise])
            M_observation_model = [[1, 0, 0]]
        else:
            M_transition = numpy.array([[1, 1], [0, 1]])
            M_process_noise = numpy.diag([self.position_noise, self.position_noise])
            M_observation_model = [[1, 0]]
        self.kf = simdkalman.KalmanFilter(
            state_transition = M_transition,
            process_noise = M_process_noise,
            observation_model = M_observation_model,
            observation_noise = self.observation_noise
        )

    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        T, H, W, C = video_array.shape

        # spatially split the video to blocks of (T, self.block_size, self.block_size, C)
        bsize = self.block_size
        denoised = numpy.empty_like(video_array)
        for h0 in tqdm(range(0, H, bsize)):
            for w0 in range(0, W, bsize):
                block = video_array[:, h0:h0+bsize, w0:w0+bsize, :]
                h, w = block.shape[1], block.shape[2]

                var_p = block.var()
                block_v = block[1:] - block[:-1]
                var_v = block_v.var()

                if self.model_acceleration:
                    var_a = (block_v[1:] - block_v[:-1]).var()
                    block_smoothed = self.kf.smooth(
                        block.transpose(1, 2, 3, 0).reshape(h*w*C, T),
                        initial_value=[block[:, 0].mean(), 0, 0],
                        initial_covariance=numpy.diag([var_p, var_v, var_a])
                    )
                else:
                    block_smoothed = self.kf.smooth(
                        block.transpose(1, 2, 3, 0).reshape(h*w*C, T),
                        initial_value=[block[:, 0].mean(), 0],
                        initial_covariance=numpy.diag([var_p, var_v])
                    )
                
                block_smoothed = block_smoothed.states.mean[..., 0].reshape(h, w, C, T).transpose(3, 0, 1, 2)
                denoised[:, h0:h0+bsize, w0:w0+bsize, :] = block_smoothed
        return denoised


class DCTThresholdingFilter(BaseFilter):
    """
    DCT thresholding denoising following:

    Guoshen Yu, and Guillermo Sapiro, 
    DCT Image Denoising: a Simple and Effective Image Denoising Algorithm, 
    Image Processing On Line, 1 (2011), pp. 292–296. 
    https://doi.org/10.5201/ipol.2011.ys-dct
    
    """
    id: str = "dct"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dtype = getattr(torch, self.dtype)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        
        T, H, W = self.kernel_size
        kernels = torch.empty(T * H * W, 1, T, H, W, dtype=self.dtype)
        with torch.no_grad():
            kt, kh, kw = torch.meshgrid(
                torch.arange(T), 
                torch.arange(H), 
                torch.arange(W), 
                indexing="ij"
            )

            for it in range(T):
                for ih in range(H):
                    for iw in range(W):
                        ct = numpy.sqrt(1/T) if it == 0 else numpy.sqrt(2/T)
                        ch = numpy.sqrt(1/H) if ih == 0 else numpy.sqrt(2/H)
                        cw = numpy.sqrt(1/W) if iw == 0 else numpy.sqrt(2/W)
                        c = ct * ch * cw

                        i = it * H * W + ih * W + iw
                        kernels[i, 0] = torch.cos((2 * kt + 1) * it * torch.pi / (2 * T)) * \
                                        torch.cos((2 * kh + 1) * ih * torch.pi / (2 * H)) * \
                                        torch.cos((2 * kw + 1) * iw * torch.pi / (2 * W)) * c
                        
            self.register_buffer("kernels", kernels.to(self.device))
            self.zo = torch.nn.Hardshrink(lambd=self.threshold)

    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        t_video = torch.from_numpy(video_array, dtype=self.dtype).to(self.device)
        
        t, h, w, c = t_video.shape
        tk, hk, wk = self.kernels.shape[2:]
        subbands = torch.nn.functional.conv3d(
            t_video.permute(3, 0, 1, 2).unsqueeze(1), 
            self.kernels, padding=(tk - 1, hk - 1, wk - 1)
        )
        subbands_filtered = self.zo(subbands)
        t_video_filtered = torch.nn.functional.conv_transpose3d(
            subbands_filtered, 
            self.kernels, 
        )[:, tk-1:-tk+1, hk-1:-hk+1, wk-1:-wk+1]
        
        video_array_filtered = t_video_filtered.permute(1, 2, 3, 0).cpu().numpy()
        return video_array_filtered

class AverageFrameBasedFilter(BaseFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        video_list = load_video_to_ndarrays(self.video_path, overlap_frames=0)
        self.average_frame = extract_average_frame(video_list)[numpy.newaxis]
        self.padding_mode = self.padding_size = None


class DarkFilter(AverageFrameBasedFilter):
    """
    subtract the dark frame extracted from the average of lens-covered video.
    """
    id: str = "dark"
    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        return video_array - self.average_frame.astype(video_array.dtype)


class FlatFilter(AverageFrameBasedFilter):
    """
    divide the video by the flat frame extracted from the average of uniform light source video.
    """
    id: str = "flat"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dark_video_path is not None:
            video_list = load_video_to_ndarrays(self.video_path, overlap_frames=0)
            self.flat_frame = self.average_frame - extract_average_frame(video_list)[numpy.newaxis]
            self.flat_frame = numpy.clip(self.flat_frame, 0, None)
        self.flat_frame = gaussian_filter(self.average_frame, sigma=self.smooth_size)
        if not self.white: # per-channel normalize
            self.flat_frame /= self.flat_frame.max(axis=(0, 1, 2), keepdims=True)
        else:
            self.flat_frame /= self.flat_frame.max()

    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        return video_array / self.flat_frame.astype(video_array.dtype)


video_filters = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and hasattr(obj, "id"):
        video_filters[obj.id] = obj
