#!/usr/bin/env python
"""
Complete SyncNet Pipeline with Preprocessing
==========================================

This script demonstrates the complete SyncNet pipeline including:
1. Video preprocessing (scene detection, face detection, tracking, cropping)
2. SyncNet analysis (audio-video synchronization)
3. Results visualization

This is equivalent to running:
- run_pipeline.py
- run_syncnet.py  
- run_visualise.py

Usage:
    python complete_syncnet_pipeline.py --videofile video.mp4 --reference my_video --data_dir ./output
"""

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path

# Import SyncNet and preprocessing functions
from syncnet_python import SyncNetInstance, preprocess_video

def run_complete_pipeline(videofile, reference, data_dir, model_path=None, 
                         facedet_scale=0.25, crop_scale=0.40, min_track=100, 
                         frame_rate=25, num_failed_det=25, min_face_size=100,
                         batch_size=20, vshift=15):
    """
    Run the complete SyncNet pipeline with preprocessing
    
    Args:
        videofile (str): Path to input video file
        reference (str): Video reference name
        data_dir (str): Output directory
        model_path (str): Path to pre-trained model
        facedet_scale (float): Face detection scale factor
        crop_scale (float): Crop scale factor
        min_track (int): Minimum track duration
        frame_rate (int): Video frame rate
        num_failed_det (int): Number of failed detections allowed
        min_face_size (int): Minimum face size in pixels
        batch_size (int): Batch size for SyncNet
        vshift (int): Video shift parameter
    
    Returns:
        dict: Complete pipeline results
    """
    
    print("🎬 Complete SyncNet Pipeline")
    print("=" * 60)
    print(f"Video: {videofile}")
    print(f"Reference: {reference}")
    print(f"Output: {data_dir}")
    print(f"Model: {model_path}")
    print("=" * 60)
    
    # Create output directories
    avi_dir = os.path.join(data_dir, 'pyavi')
    tmp_dir = os.path.join(data_dir, 'pytmp')
    work_dir = os.path.join(data_dir, 'pywork')
    crop_dir = os.path.join(data_dir, 'pycrop')
    frames_dir = os.path.join(data_dir, 'pyframes')
    
    for dir_path in [avi_dir, tmp_dir, work_dir, crop_dir, frames_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Configuration object for preprocessing
    class PreprocessConfig:
        def __init__(self):
            self.videofile = videofile
            self.reference = reference
            self.avi_dir = avi_dir
            self.tmp_dir = tmp_dir
            self.work_dir = work_dir
            self.crop_dir = crop_dir
            self.frames_dir = frames_dir
            self.facedet_scale = facedet_scale
            self.crop_scale = crop_scale
            self.min_track = min_track
            self.frame_rate = frame_rate
            self.num_failed_det = num_failed_det
            self.min_face_size = min_face_size
    
    preprocess_opt = PreprocessConfig()
    
    # Step 1: Video Preprocessing
    print("\n🎭 Step 1: Video Preprocessing")
    print("-" * 40)
    print("This includes:")
    print("  • Video conversion and frame extraction")
    print("  • Scene detection")
    print("  • Face detection and tracking")
    print("  • Face cropping and preparation")
    
    preprocessing_results = preprocess_video(preprocess_opt)
    
    if preprocessing_results['status'] != 'success':
        print(f"❌ Preprocessing failed: {preprocessing_results.get('error', 'Unknown error')}")
        return {
            'status': 'error',
            'error': 'Preprocessing failed',
            'details': preprocessing_results
        }
    
    print(f"✅ Preprocessing completed!")
    print(f"   • Detected {len(preprocessing_results['scenes'])} scenes")
    print(f"   • Found {len(preprocessing_results['tracks'])} face tracks")
    
    # Step 2: SyncNet Analysis
    print("\n🎵 Step 2: SyncNet Analysis")
    print("-" * 40)
    
    try:
        # Initialize SyncNet
        print("🚀 Initializing SyncNet...")
        syncnet = SyncNetInstance()
        
        # Load pre-trained model
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            syncnet.loadParameters(model_path)
        else:
            print("⚠️  No pre-trained model found. Using random weights.")
        
        # Configure SyncNet parameters
        class SyncNetConfig:
            def __init__(self):
                self.tmp_dir = tmp_dir
                self.reference = reference
                self.batch_size = batch_size
                self.vshift = vshift
        
        syncnet_opt = SyncNetConfig()
        
        # Run SyncNet analysis
        print("🔄 Running SyncNet analysis...")
        offset, confidence, distances = syncnet.evaluate(syncnet_opt, videofile)
        
        # Save SyncNet results
        results_file = os.path.join(work_dir, reference, 'offsets.txt')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write(f"SyncNet Analysis Results\n")
            f.write(f"======================\n\n")
            f.write(f"Video: {videofile}\n")
            f.write(f"Reference: {reference}\n")
            f.write(f"Audio-Video Offset: {offset} frames\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ SyncNet analysis completed!")
        print(f"   • Audio-Video Offset: {offset} frames")
        print(f"   • Confidence: {confidence:.3f}")
        print(f"   • Results saved to: {results_file}")
        
        # Step 3: Results Summary
        print("\n📊 Step 3: Results Summary")
        print("-" * 40)
        
        # Interpretation
        if confidence > 10:
            sync_status = "🟢 Excellent synchronization"
        elif confidence > 5:
            sync_status = "🟡 Good synchronization"
        else:
            sync_status = "🔴 Poor synchronization - may need manual adjustment"
        
        print(f"Sync Status: {sync_status}")
        print(f"Processing completed successfully!")
        
        return {
            'status': 'success',
            'video': videofile,
            'reference': reference,
            'offset': float(offset),
            'confidence': float(confidence),
            'distances': distances,
            'preprocessing': preprocessing_results,
            'results_file': results_file,
            'sync_status': sync_status
        }
        
    except Exception as e:
        print(f"❌ SyncNet analysis failed: {str(e)}")
        return {
            'status': 'error',
            'error': 'SyncNet analysis failed',
            'details': str(e),
            'preprocessing': preprocessing_results
        }

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='Complete SyncNet Pipeline with Preprocessing')
    parser.add_argument('--videofile', type=str, required=True,
                       help='Input video file')
    parser.add_argument('--reference', type=str, required=True,
                       help='Video reference name')
    parser.add_argument('--data_dir', type=str, default='./syncnet_output',
                       help='Output data directory')
    parser.add_argument('--model_path', type=str, default='syncnet_model.pth',
                       help='Path to pre-trained model')
    
    # Preprocessing parameters
    parser.add_argument('--facedet_scale', type=float, default=0.25,
                       help='Face detection scale factor')
    parser.add_argument('--crop_scale', type=float, default=0.40,
                       help='Crop scale factor')
    parser.add_argument('--min_track', type=int, default=100,
                       help='Minimum track duration')
    parser.add_argument('--frame_rate', type=int, default=25,
                       help='Video frame rate')
    parser.add_argument('--num_failed_det', type=int, default=25,
                       help='Number of failed detections allowed')
    parser.add_argument('--min_face_size', type=int, default=100,
                       help='Minimum face size in pixels')
    
    # SyncNet parameters
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size for SyncNet')
    parser.add_argument('--vshift', type=int, default=15,
                       help='Video shift parameter')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.videofile):
        print(f"❌ Error: Video file not found: {args.videofile}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Run the complete pipeline
    start_time = time.time()
    results = run_complete_pipeline(
        videofile=args.videofile,
        reference=args.reference,
        data_dir=args.data_dir,
        model_path=args.model_path,
        facedet_scale=args.facedet_scale,
        crop_scale=args.crop_scale,
        min_track=args.min_track,
        frame_rate=args.frame_rate,
        num_failed_det=args.num_failed_det,
        min_face_size=args.min_face_size,
        batch_size=args.batch_size,
        vshift=args.vshift
    )
    
    # Print final results
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)
    
    if results['status'] == 'success':
        print(f"✅ Video: {results['video']}")
        print(f"🏷️  Reference: {results['reference']}")
        print(f"📊 Audio-Video Offset: {results['offset']} frames")
        print(f"🎯 Confidence: {results['confidence']:.3f}")
        print(f"📈 Sync Status: {results['sync_status']}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"📁 Results saved to: {args.data_dir}")
        
        if 'preprocessing' in results:
            prep = results['preprocessing']
            print(f"🎭 Scenes detected: {len(prep.get('scenes', []))}")
            print(f"👤 Face tracks: {len(prep.get('tracks', []))}")
        
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        if 'details' in results:
            print(f"Details: {results['details']}")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
