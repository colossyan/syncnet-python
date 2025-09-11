#!/usr/bin/env python
"""
Simple SyncNet Usage Example
===========================

This shows how to use the simple pipeline functions in your code.
"""

from syncnet_python import run_pipeline, run_syncnet, run_visualise, run_complete_pipeline

def main():
    # Configuration
    videofile = "video.mp4"
    reference = "my_video"
    data_dir = "./output"
    model_path = "syncnet_model.pth"
    
    print("🎬 SyncNet Pipeline Example")
    print("=" * 40)
    
    # Method 1: Run each step separately
    print("\n📋 Method 1: Step by step")
    print("-" * 30)
    
    # Step 1: Preprocessing
    print("Step 1: Preprocessing...")
    pipeline_results = run_pipeline(videofile, reference, data_dir)
    
    if pipeline_results['status'] == 'success':
        print(f"✅ Preprocessing completed: {len(pipeline_results['tracks'])} tracks found")
        
        # Step 2: SyncNet analysis
        print("Step 2: SyncNet analysis...")
        syncnet_results = run_syncnet(videofile, reference, data_dir, model_path)
        
        if syncnet_results['status'] == 'success':
            print(f"✅ SyncNet completed: Offset={syncnet_results['offset']}, Confidence={syncnet_results['confidence']:.3f}")
            
            # Step 3: Visualization
            print("Step 3: Visualization...")
            visualise_results = run_visualise(videofile, reference, data_dir)
            
            if visualise_results['status'] == 'success':
                print("✅ Visualization completed")
            else:
                print(f"⚠️  Visualization failed: {visualise_results.get('error', 'Unknown error')}")
        else:
            print(f"❌ SyncNet failed: {syncnet_results.get('error', 'Unknown error')}")
    else:
        print(f"❌ Preprocessing failed: {pipeline_results.get('error', 'Unknown error')}")
    
    # Method 2: Run complete pipeline at once
    print("\n📋 Method 2: Complete pipeline")
    print("-" * 30)
    
    complete_results = run_complete_pipeline(
        videofile=videofile,
        reference=reference + "_complete",
        data_dir=data_dir,
        model_path=model_path
    )
    
    if complete_results['status'] == 'success':
        print("✅ Complete pipeline finished successfully!")
        syncnet = complete_results['syncnet']
        print(f"   Offset: {syncnet['offset']} frames")
        print(f"   Confidence: {syncnet['confidence']:.3f}")
    else:
        print(f"❌ Complete pipeline failed: {complete_results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
