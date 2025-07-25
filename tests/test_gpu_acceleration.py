#!/usr/bin/env python3
"""
Test script to verify GPU acceleration with Apple Metal Performance Shaders.
"""

import time
import torch
from heimdall.core.detection import ObjectDetector

def test_device_availability():
    """Test device availability and performance."""
    print("=== Device Availability Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

def benchmark_inference_speed(video_path: str = "test_data/videoplayback.mp4"):
    """Benchmark inference speed with CPU vs GPU."""
    if not video_path:
        print("No video file provided, skipping benchmark")
        return
    
    print("=== Inference Speed Benchmark ===")
    
    # Test CPU performance
    print("Testing CPU performance...")
    detector_cpu = ObjectDetector(
        model_name="yolov8n-640",  # Use smaller model for faster testing
        use_gpu=False,
        prefer_ultralytics=True
    )
    
    start_time = time.time()
    detector_cpu.start_detection(video_path)
    time.sleep(5)  # Run for 5 seconds
    detector_cpu.stop_detection()
    cpu_time = time.time() - start_time
    print(f"CPU inference completed in {cpu_time:.2f} seconds")
    
    # Test GPU performance
    print("\nTesting GPU performance...")
    detector_gpu = ObjectDetector(
        model_name="yolov8n-640",  # Use smaller model for faster testing
        use_gpu=True,
        prefer_ultralytics=True
    )
    
    start_time = time.time()
    detector_gpu.start_detection(video_path)
    time.sleep(5)  # Run for 5 seconds
    detector_gpu.stop_detection()
    gpu_time = time.time() - start_time
    print(f"GPU inference completed in {gpu_time:.2f} seconds")
    
    if cpu_time > 0 and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1.1:
            print("✅ GPU acceleration is working!")
        else:
            print("⚠️  GPU acceleration may not be optimal")
    
    print()

def test_basic_functionality():
    """Test basic detection functionality."""
    print("=== Basic Functionality Test ===")
    
    try:
        detector = ObjectDetector(use_gpu=True, prefer_ultralytics=True)
        print("✅ ObjectDetector initialized successfully")
        print(f"   Using device: {detector.device}")
        print(f"   Using Ultralytics: {detector.use_ultralytics}")
    except Exception as e:
        print(f"❌ Failed to initialize ObjectDetector: {e}")
        return
    
    print()

if __name__ == "__main__":
    print("Apple Metal GPU Acceleration Test")
    print("=" * 40)
    
    test_device_availability()
    test_basic_functionality()
    
    # Only run benchmark if video file exists
    import os
    video_path = "test_data/videoplayback.mp4"
    if os.path.exists(video_path):
        benchmark_inference_speed(video_path)
    else:
        print(f"Video file {video_path} not found, skipping benchmark")
        print("You can run the benchmark by providing a video file path")