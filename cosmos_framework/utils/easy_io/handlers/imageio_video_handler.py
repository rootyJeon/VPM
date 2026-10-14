# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import IO, Any, Dict, Tuple

import imageio
import imageio.v3 as iio_v3
import numpy as np
import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.easy_io.handlers.base import BaseFileHandler


class ImageioVideoHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(
        self, file: IO[bytes], format: str = "mp4", mode: str = "rgb", **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load video from a file-like object using imageio.v3 with specified format and color mode.

        Parameters:
            file (IO[bytes]): A file-like object containing video data.
            format (str): Format of the video file (default 'mp4').
            mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

        Returns:
            tuple: A tuple containing an array of video frames and metadata about the video.
        """
        file.seek(0)

        plugin = kwargs.pop("plugin", "pyav")

        video_frames = iio_v3.imread(file, plugin=plugin, **kwargs)

        if mode == "gray":
            import cv2

            if len(video_frames.shape) == 4:
                gray_frames = []
                for frame in video_frames:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    gray_frame = np.expand_dims(gray_frame, axis=2)
                    gray_frames.append(gray_frame)
                video_frames = np.array(gray_frames)

        file.seek(0)
        metadata = self._extract_metadata(file, plugin=plugin)

        return video_frames, metadata

    def _extract_metadata(self, file: IO[bytes], plugin: str = "pyav") -> Dict[str, Any]:
        """
        Extract metadata from a video file.

        Parameters:
            file (IO[bytes]): File-like object containing video data.
            plugin (str): Plugin to use for reading.

        Returns:
            dict: Video metadata.
        """
        try:
            metadata = iio_v3.immeta(file, plugin=plugin)

            if "fps" not in metadata and "duration" in metadata:
                file.seek(0)
                first_frame = iio_v3.imread(file, plugin=plugin, index=0)
                metadata["size"] = first_frame.shape[1::-1]
                metadata["source_size"] = metadata["size"]

                metadata["plugin"] = plugin
                if "codec" not in metadata:
                    metadata["codec"] = "unknown"
                if "pix_fmt" not in metadata:
                    metadata["pix_fmt"] = "unknown"

                if "fps" in metadata and "duration" in metadata:
                    metadata["nframes"] = int(metadata["fps"] * metadata["duration"])
                else:
                    metadata["nframes"] = float("inf")

            return metadata

        except Exception as e:
            return {
                "plugin": plugin,
                "nframes": float("inf"),
                "codec": "unknown",
                "fps": 30.0,
                "duration": 0,
                "size": (0, 0),
            }

    def dump_to_fileobj(
        self,
        obj: np.ndarray | torch.Tensor,
        file: IO[bytes],
        format: str = "mp4",
        fps: int = 17,
        quality: int = 5,
        ffmpeg_params=None,
        **kwargs,
    ):
        """
        Save an array of video frames to a file-like object using imageio.

        Parameters:
            obj (Union[np.ndarray, torch.Tensor]): An array of frames to be saved as video.
            file (IO[bytes]): A file-like object to which the video data will be written.
            format (str): Format of the video file (default 'mp4').
            fps (int): Frames per second of the output video (default 17).
            quality (int): Quality of the video (0-10, default 5).
            ffmpeg_params (list): Additional parameters to pass to ffmpeg.

        """
        if isinstance(obj, torch.Tensor):
            assert obj.dtype == torch.uint8, "Tensor must be of type uint8"
            obj = obj.cpu().numpy()
        h, w = obj.shape[1:-1]

        default_ffmpeg_params = ["-s", f"{w}x{h}"]

        final_ffmpeg_params = ffmpeg_params if ffmpeg_params is not None else default_ffmpeg_params

        mimsave_kwargs = {
            "fps": fps,
            "quality": quality,
            "macro_block_size": 1,
            "ffmpeg_params": final_ffmpeg_params,
            "output_params": ["-f", "mp4"],
        }
        mimsave_kwargs.update(kwargs)
        log.debug(f"mimsave_kwargs: {mimsave_kwargs}")

        imageio.mimsave(file, obj, format, **mimsave_kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
