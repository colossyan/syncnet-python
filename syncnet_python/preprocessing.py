#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
SyncNet Preprocessing Module
===========================

This module contains the preprocessing functions from run_pipeline.py
that are essential for SyncNet to work properly:
- Scene detection
- Face detection and tracking
- Video cropping and preparation
"""

import os
import time
import pickle
import subprocess
import glob
import cv2
import numpy as np
from shutil import rmtree
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

try:
    import scenedetect
    from scenedetect.video_manager import VideoManager
    from scenedetect.scene_manager import SceneManager
    from scenedetect.frame_timecode import FrameTimecode
    from scenedetect.stats_manager import StatsManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    from detectors import S3FD
    FACE_DETECTOR_AVAILABLE = True
except ImportError:
    FACE_DETECTOR_AVAILABLE = False

def bb_intersection_over_union(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def track_shot(opt, scenefaces):
    """Track faces across frames in a shot"""
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    
    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        
        if track == []:
            break
        elif len(track) > opt.min_track:
            framenum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            
            frame_i = np.arange(framenum[0], framenum[-1] + 1)
            
            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), 
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    
    return tracks

def crop_video(opt, track, cropfile):
    """Crop video based on face tracking"""
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, '*.jpg'))
    flist.sort()
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile + 't.avi', fourcc, opt.frame_rate, (224, 224))
    
    dets = {'x': [], 'y': [], 's': []}
    
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)  # crop center x
        dets['x'].append((det[0] + det[2]) / 2)  # crop center y
    
    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    
    for fidx, frame in enumerate(track['frame']):
        cs = opt.crop_scale
        
        bs = dets['s'][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        
        image = cv2.imread(flist[frame])
        
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X
        
        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), 
                     int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        
        vOut.write(cv2.resize(face, (224, 224)))
    
    audiotmp = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
    audiostart = (track['frame'][0]) / opt.frame_rate
    audioend = (track['frame'][-1]) / opt.frame_rate
    
    command = ("ffmpeg -y -i %s -ss %.3f -t %.3f %s" % (audiotmp, audiostart, audioend - audiostart, cropfile + '.wav'))
    output = subprocess.call(command, shell=True, stdout=None)
    
    vOut.release()
    
    return {'track': track, 'duration': audioend - audiostart}

def inference_video(opt):
    """Run face detection on video frames"""
    if not FACE_DETECTOR_AVAILABLE:
        raise ImportError("Face detector not available. Please install the detectors module.")
    
    DET = S3FD(device='cuda')
    
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, '*.jpg'))
    flist.sort()
    
    dets = []
    
    for fidx, fname in enumerate(flist):
        start_time = time.time()
        
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
        
        elapsed_time = time.time() - start_time
        print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), 
                                            fidx, len(dets[-1]), (1 / elapsed_time)))
    
    savepath = os.path.join(opt.work_dir, opt.reference, 'faces.pckl')
    
    with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)
    
    return dets

def scene_detect(opt):
    """Detect scenes in the video"""
    if not SCENEDETECT_AVAILABLE:
        raise ImportError("Scene detection not available. Please install scenedetect.")
    
    video_manager = VideoManager([os.path.join(opt.avi_dir, opt.reference, 'video.avi')])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    
    video_manager.set_downscale_factor()
    video_manager.start()
    
    scene_manager.detect_scenes(frame_source=video_manager)
    
    scene_list = scene_manager.get_scene_list(base_timecode)
    
    savepath = os.path.join(opt.work_dir, opt.reference, 'scene.pckl')
    
    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    
    with open(savepath, 'wb') as fil:
        pickle.dump(scene_list, fil)
    
    print('%s - scenes detected %d' % (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), len(scene_list)))
    
    return scene_list

def preprocess_video(opt):
    """
    Complete video preprocessing pipeline
    
    Args:
        opt: Configuration object with video paths and parameters
    
    Returns:
        dict: Results including tracks and processing status
    """
    print("🎬 Starting video preprocessing pipeline...")
    
    # Clean up existing directories
    for dir_path in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        if os.path.exists(os.path.join(dir_path, opt.reference)):
            rmtree(os.path.join(dir_path, opt.reference))
    
    # Create new directories
    for dir_path in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        os.makedirs(os.path.join(dir_path, opt.reference))
    
    try:
        # Step 1: Convert video and extract frames
        print("🔄 Converting video and extracting frames...")
        command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % 
                  (opt.videofile, os.path.join(opt.avi_dir, opt.reference, 'video.avi')))
        subprocess.call(command, shell=True, stdout=None)
        
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % 
                  (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), 
                   os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')))
        subprocess.call(command, shell=True, stdout=None)
        
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % 
                  (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), 
                   os.path.join(opt.avi_dir, opt.reference, 'audio.wav')))
        subprocess.call(command, shell=True, stdout=None)
        
        # Step 2: Face detection
        print("👤 Running face detection...")
        faces = inference_video(opt)
        
        # Step 3: Scene detection
        print("🎭 Detecting scenes...")
        scene = scene_detect(opt)
        
        # Step 4: Face tracking
        print("🔍 Tracking faces...")
        alltracks = []
        vidtracks = []
        
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
                alltracks.extend(track_shot(opt, faces[shot[0].frame_num:shot[1].frame_num]))
        
        # Step 5: Face track cropping
        print("✂️  Cropping face tracks...")
        for ii, track in enumerate(alltracks):
            vidtracks.append(crop_video(opt, track, os.path.join(opt.crop_dir, opt.reference, '%05d' % ii)))
        
        # Step 6: Save results
        savepath = os.path.join(opt.work_dir, opt.reference, 'tracks.pckl')
        with open(savepath, 'wb') as fil:
            pickle.dump(vidtracks, fil)
        
        # Cleanup
        rmtree(os.path.join(opt.tmp_dir, opt.reference))
        
        print("✅ Preprocessing completed successfully!")
        return {
            'status': 'success',
            'tracks': vidtracks,
            'faces': faces,
            'scenes': scene,
            'tracks_file': savepath
        }
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }
