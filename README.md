# SyncNet

> **Fork Notice**: This is a fork of the original [SyncNet repository](https://github.com/joonson/syncnet_python) by Joon Son Chung. This fork is maintained by [Colossyan](https://github.com/colossyan).

This repository contains the demo for the audio-to-video synchronisation network (SyncNet). This network can be used for audio-visual synchronisation tasks including: 
1. Removing temporal lags between the audio and visual streams in a video;
2. Determining who is speaking amongst multiple faces in a video. 

Please cite the paper below if you make use of the software. 

## Installation

### Option 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/colossyan/syncnet-python.git
```

### Option 2: Install from source
```bash
git clone https://github.com/colossyan/syncnet-python.git
cd syncnet-python
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

In addition, `ffmpeg` is required.

## Usage

### As a Python Package

```python
from syncnet_python import SyncNetInstance

# Initialize SyncNet
syncnet = SyncNetInstance()

# Load pre-trained model
syncnet.loadParameters('path/to/pretrained_model.pth')

# Evaluate a video
class Args:
    def __init__(self):
        self.tmp_dir = '/tmp/syncnet'
        self.reference = 'test_video'
        self.batch_size = 20
        self.vshift = 10

opt = Args()
offset, conf, dists = syncnet.evaluate(opt, 'path/to/video.mp4')
print(f"Audio-Video offset: {offset}")
print(f"Confidence: {conf}")
```

### Command Line Usage

SyncNet demo:
```bash
python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

Check that this script returns:
```
AV offset:      3 
Min dist:       5.353
Confidence:     10.021
```

Full pipeline:
```bash
sh download_model.sh
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_visualise.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
```

Outputs:
```
$DATA_DIR/pycrop/$REFERENCE/*.avi - cropped face tracks
$DATA_DIR/pywork/$REFERENCE/offsets.txt - audio-video offset values
$DATA_DIR/pyavi/$REFERENCE/video_out.avi - output video (as shown below)
```
<p align="center">
  <img src="img/ex1.jpg" width="45%"/>
  <img src="img/ex2.jpg" width="45%"/>
</p>

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
