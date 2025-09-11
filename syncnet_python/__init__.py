"""
SyncNet: Audio-to-video synchronisation network for lip sync and speaker identification.

This package provides tools for:
1. Removing temporal lags between the audio and visual streams in a video
2. Determining who is speaking amongst multiple faces in a video
3. Complete video preprocessing pipeline including face detection and tracking

Simple usage:
    from syncnet_python import run_pipeline, run_syncnet, run_visualise
    
    # Run preprocessing
    pipeline_results = run_pipeline('video.mp4', 'my_video', './output')
    
    # Run SyncNet analysis
    syncnet_results = run_syncnet('video.mp4', 'my_video', './output')
    
    # Run visualization
    visualise_results = run_visualise('video.mp4', 'my_video', './output')
"""

from .SyncNetInstance import SyncNetInstance
from .SyncNetModel import S, save, load
from .pipeline_functions import (
    run_pipeline,
    run_syncnet,
    run_visualise,
    run_complete_pipeline
)

__version__ = "1.0.0"
__author__ = "Colossyan"
__email__ = "info@colossyan.com"

__all__ = [
    "SyncNetInstance",
    "S", 
    "save",
    "load",
    "run_pipeline",
    "run_syncnet", 
    "run_visualise",
    "run_complete_pipeline"
]
