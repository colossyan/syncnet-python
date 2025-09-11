#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SyncNet Python Package

Audio-to-video synchronisation network for lip sync and speaker identification.
"""

__version__ = "1.0.0"
__author__ = "Colossyan"
__email__ = "ahmad@colossyan.com"

from .SyncNetInstance import SyncNetInstance
from .syncnet_pipeline import SyncNetPipeline

__all__ = [
    "SyncNetInstance",
    "SyncNetPipeline",
]
