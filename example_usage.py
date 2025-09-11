#!/usr/bin/env python
"""
Example usage of the syncnet-python package.
This demonstrates how to use SyncNet after installing via pip.
"""

import torch
from syncnet_python import SyncNetInstance

def main():
    # Initialize SyncNet
    syncnet = SyncNetInstance()
    
    # Load pre-trained model (you would need to download this)
    # syncnet.loadParameters('path/to/pretrained_model.pth')
    
    print("SyncNet initialized successfully!")
    print("You can now use syncnet.evaluate() or syncnet.extract_feature() methods")
    
    # Example usage (requires a video file and proper setup):
    # class Args:
    #     def __init__(self):
    #         self.tmp_dir = '/tmp/syncnet'
    #         self.reference = 'test_video'
    #         self.batch_size = 20
    #         self.vshift = 10
    # 
    # opt = Args()
    # offset, conf, dists = syncnet.evaluate(opt, 'path/to/video.mp4')
    # print(f"Audio-Video offset: {offset}")
    # print(f"Confidence: {conf}")

if __name__ == "__main__":
    main()
