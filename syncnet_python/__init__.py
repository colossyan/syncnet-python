"""
SyncNet: Audio-to-video synchronisation network for lip sync and speaker identification.

This package provides tools for:
1. Removing temporal lags between the audio and visual streams in a video
2. Determining who is speaking amongst multiple faces in a video
"""

from .SyncNetInstance import SyncNetInstance
from .SyncNetModel import S, save, load

__version__ = "1.0.0"
__author__ = "Colossyan"
__email__ = "ahmad@colossyan.com"

__all__ = [
    "SyncNetInstance",
    "S", 
    "save",
    "load"
]
