#!/usr/bin/python
#-*- coding: utf-8 -*-
import time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
from detectors import S3FD
from syncnet_python.SyncNetInstance import SyncNetInstance

# ========== PARSE ARGS ==========

parser = argparse.ArgumentParser(description="FaceTracker and SyncNet")
parser.add_argument('--data_dir', type=str, default='data/work', help='Output directory')
parser.add_argument('--videofile', type=str, default='', help='Input video file')
parser.add_argument('--reference', type=str, default='', help='Video reference')
parser.add_argument('--facedet_scale', type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--crop_scale', type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--min_track', type=int, default=100, help='Minimum facetrack duration')
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25, help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--min_face_size', type=int, default=100, help='Minimum face size in pixels')
parser.add_argument('--s3fd_model', type=str, default='detectors/s3fd/weights/sfd_face.pth', help='Path to S3FD model file')
parser.add_argument('--syncnet_model', type=str, default='/data/ahmad/syncnet_python/data/syncnet_v2.model', help='Path to SyncNet model file')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for SyncNet evaluation')
parser.add_argument('--vshift', type=int, default=15, help='Video shift for SyncNet evaluation')
opt = parser.parse_args()

# Set directory attributes
setattr(opt, 'avi_dir', os.path.join(opt.data_dir, 'pyavi'))
setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))
setattr(opt, 'frames_dir', os.path.join(opt.data_dir, 'pyframes'))  # Ensure frames_dir is set

# Validate input arguments
if not os.path.isfile(opt.videofile):
    raise FileNotFoundError(f"Video file not found: {opt.videofile}")
if not opt.reference:
    raise ValueError("Reference name must be provided")
if not os.path.isfile(opt.s3fd_model):
    raise FileNotFoundError(f"S3FD model file not found: {opt.s3fd_model}")
if not os.path.isfile(opt.syncnet_model):
    raise FileNotFoundError(f"SyncNet model file not found: {opt.syncnet_model}")

# ========== IOU FUNCTION ==========

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ========== FACE TRACKING ==========

def track_shot(opt, scenefaces):
    iouThres = 0.5
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
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    return tracks

# ========== VIDEO CROP AND SAVE ==========

def crop_video(opt, track, cropfile):
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, '*.jpg'))
    flist.sort()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile + 't.avi', fourcc, opt.frame_rate, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = opt.crop_scale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audiotmp = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
    audiostart = (track['frame'][0]) / opt.frame_rate
    audioend = (track['frame'][-1] + 1) / opt.frame_rate
    vOut.release()
    command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (
        os.path.join(opt.avi_dir, opt.reference, 'audio.wav'), audiostart, audioend, audiotmp))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        pdb.set_trace()
    sample_rate, audio = wavfile.read(audiotmp)
    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile, audiotmp, cropfile))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        pdb.set_trace()
    print('Written %s' % cropfile)
    os.remove(cropfile + 't.avi')
    print('Mean pos: x %.2f y %.2f s %.2f' % (np.mean(dets['x']), np.mean(dets['y']), np.mean(dets['s'])))
    return {'track': track, 'proc_track': dets}

# ========== FACE DETECTION ==========

def inference_video(opt):
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
        print('%s-%05d; %d dets; %.2f Hz' % (
            os.path.join(opt.avi_dir, opt.reference, 'video.avi'), fidx, len(dets[-1]), (1 / elapsed_time)))
    savepath = os.path.join(opt.work_dir, opt.reference, 'faces.pckl')
    with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets

# ========== SCENE DETECTION ==========

def scene_detect(opt):
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

# ========== STAGE I: MODULAR FUNCTIONS ==========

def delete_existing_directories(opt):
    """Delete existing directories if they exist."""
    for directory in [
        os.path.join(opt.work_dir, opt.reference),
        os.path.join(opt.crop_dir, opt.reference),
        os.path.join(opt.avi_dir, opt.reference),
        os.path.join(opt.frames_dir, opt.reference),
        os.path.join(opt.tmp_dir, opt.reference)
    ]:
        if os.path.exists(directory):
            rmtree(directory)

def create_new_directories(opt):
    """Create new directories for processing."""
    for directory in [
        os.path.join(opt.work_dir, opt.reference),
        os.path.join(opt.crop_dir, opt.reference),
        os.path.join(opt.avi_dir, opt.reference),
        os.path.join(opt.frames_dir, opt.reference),
        os.path.join(opt.tmp_dir, opt.reference)
    ]:
        os.makedirs(directory)

def convert_video_and_extract_frames(opt):
    """Convert video to AVI and extract frames and audio."""
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (
        opt.videofile, os.path.join(opt.avi_dir, opt.reference, 'video.avi')))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        raise RuntimeError("Failed to convert video to AVI")

    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (
        os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
        os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        raise RuntimeError("Failed to extract frames")

    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (
        os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
        os.path.join(opt.avi_dir, opt.reference, 'audio.wav')))
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        raise RuntimeError("Failed to extract audio")

def perform_face_detection(opt):
    """Perform face detection on video frames."""
    return inference_video(opt)

def perform_scene_detection(opt):
    """Perform scene detection on the video."""
    return scene_detect(opt)

def perform_face_tracking(opt, scenes, faces):
    """Perform face tracking across scenes."""
    alltracks = []
    for shot in scenes:
        if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
            alltracks.extend(track_shot(opt, faces[shot[0].frame_num:shot[1].frame_num]))
    return alltracks

def perform_face_track_cropping(opt, tracks):
    """Crop video based on face tracks."""
    vidtracks = []
    for ii, track in enumerate(tracks):
        vidtracks.append(crop_video(opt, track, os.path.join(opt.crop_dir, opt.reference, '%05d' % ii)))
    return vidtracks

def save_stage1_results(opt, vidtracks):
    """Save the tracking results from Stage I."""
    savepath = os.path.join(opt.work_dir, opt.reference, 'tracks.pckl')
    with open(savepath, 'wb') as fil:
        pickle.dump(vidtracks, fil)

def cleanup_temp_directory(opt):
    """Clean up temporary directory."""
    rmtree(os.path.join(opt.tmp_dir, opt.reference))

# ========== STAGE II: MODULAR FUNCTIONS ==========
def load_syncnet_model(opt):
    """Load the SyncNet model."""
    s = SyncNetInstance()
    s.loadParameters(opt.syncnet_model)  # Fixed to use opt.syncnet_model
    print(f"Model {opt.syncnet_model} loaded.")
    return s

def evaluate_syncnet(opt, syncnet_instance):
    """Evaluate cropped videos using SyncNet and collect distances."""
    flist = glob.glob(os.path.join(opt.crop_dir, opt.reference, '0*.avi'))
    flist.sort()
    if not flist:
        raise FileNotFoundError(f"No cropped video files found in {os.path.join(opt.crop_dir, opt.reference)}")
    dists = []
    for idx, fname in enumerate(flist):
        offset, conf, dist = syncnet_instance.evaluate(opt, videofile=fname)
        dists.append(dist)
    return dists

def save_stage2_results(opt, dists):
    """Save the SyncNet evaluation results."""
    savepath = os.path.join(opt.work_dir, opt.reference, 'activesd.pckl')
    with open(savepath, 'wb') as fil:
        pickle.dump(dists, fil)
# ========== MAIN ENDPOINT FUNCTION ==========
def process_video(opt):
    """Main endpoint function to process the video through Stage I and Stage II."""
    try:
        # Stage I: Face Tracking and Cropping
        print("Starting Stage I: Face Tracking and Cropping...")
        
        # Step 1: Delete existing directories
        delete_existing_directories(opt)
        
        # Step 2: Create new directories
        create_new_directories(opt)
        
        # Step 3: Convert video and extract frames
        convert_video_and_extract_frames(opt)
        
        # Step 4: Face detection
        faces = perform_face_detection(opt)
        
        # Step 5: Scene detection
        scenes = perform_scene_detection(opt)
        
        # Step 6: Face tracking
        tracks = perform_face_tracking(opt, scenes, faces)
        
        # Step 7: Face track cropping
        vidtracks = perform_face_track_cropping(opt, tracks)
        
        # Step 8: Save Stage I results
        save_stage1_results(opt, vidtracks)
        
        # Step 9: Clean up temporary directory
        cleanup_temp_directory(opt)
        
        print("Stage I completed successfully.")

        # Stage II: SyncNet Evaluation
        print("Starting Stage II: SyncNet Evaluation...")
        
        # Step 10: Load SyncNet model
        syncnet_instance = load_syncnet_model(opt)
        
        # Step 11: Evaluate SyncNet on cropped videos
        dists = evaluate_syncnet(opt, syncnet_instance)
        
        # Step 12: Save Stage II results
        save_stage2_results(opt, dists)
        
        print("Stage II completed successfully.")
        print("Video processing completed successfully.")
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        raise

# ========== EXECUTE MAIN ==========

if __name__ == "__main__":
    process_video(opt)