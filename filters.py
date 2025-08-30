import sys
import inspect
from abc import ABC, abstractmethod
import numpy
from scipy.signal import wiener
from typing import Iterable

from utils import logger


class BaseFilter(ABC):
    """
    Base class for all filters. Handel padding and de-padding only.
    """
    def __init__(self, **kwargs):
        if "padding_size" not in kwargs or \
            not isinstance(kwargs["padding_size"], Iterable) or \
                len(kwargs["padding_size"]) != 4:
            raise ValueError("padding_size must be specified")
        if "padding_mode" not in kwargs:
            kwargs["padding_mode"] = "reflect" # by default use reflect, the value will not be checked
            
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def filter(self, video_array: numpy.ndarray) -> numpy.ndarray:
        pass

    def __call__(self, video_array: numpy.ndarray) -> numpy.ndarray:
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

class WienerFilter(BaseFilter):
    id: str = "wiener"
    def __init__(self, **kwargs):
        if "noise" not in kwargs:
            kwargs["noise"] = "auto"
        elif kwargs["noise"] not in ["auto", "pixel-wise"]:
            raise ValueError("noise must be \"auto\" or \"pixel-wise\"")
        if kwargs["noise"] == "pixel-wise":
            logger.warning("In pixel-wise mode, only the first dimension of window_size is used")
            if "var_scale" not in kwargs:
                kwargs["var_scale"] = numpy.sqrt(2)
        super().__init__(**kwargs)
    
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

            logger.info("Pix-wise denoising with wiener filter ...")
            # pix-wise denoise with wiener filter
            denoised = numpy.zeros_like(video_array)
            for h in tqdm(range(H)):
                for w in range(W):
                    noise_var = var_map[h, w] * self.var_scale # empirical factor median --> mean
                    if noise_var > 0:
                        denoised[:, h, w, :] = scipy.signal.wiener(video_array[:, h, w, :], (t_window_size, 1), noise_var)
                    else:
                        denoised[:, h, w, :] = video_array[:, h, w, :]
            return denoised

# Load all the classes from filters.py who has id attribute, and make the id as key, the class as corresponding value
video_filters = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and hasattr(obj, "id"):
        video_filters[obj.id] = obj
