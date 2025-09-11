"""
SyncNet: Audio-to-video synchronisation network for lip sync and speaker identification.

This package provides tools for:
1. Removing temporal lags between the audio and visual streams in a video
2. Determining who is speaking amongst multiple faces in a video
3. Complete video preprocessing pipeline including face detection and tracking
"""

from .SyncNetInstance import SyncNetInstance
from .SyncNetModel import S, save, load
from .preprocessing import (
    preprocess_video,
    inference_video,
    scene_detect,
    track_shot,
    crop_video,
    bb_intersection_over_union
)

__version__ = "1.0.0"
__author__ = "Colossyan"
__email__ = "info@colossyan.com"

__all__ = [
    "SyncNetInstance",
    "S", 
    "save",
    "load",
    "preprocess_video",
    "inference_video",
    "scene_detect",
    "track_shot",
    "crop_video",
    "bb_intersection_over_union"
]
