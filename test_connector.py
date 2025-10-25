"""
Test script for StutterDetector connector
Tests the connector with a sample audio file
"""

import sys
import os

# Add App directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'App'))

from connector import StutterDetector

def test_connector():
    """Test the connector with a sample audio file"""
    
    print("=" * 60)
    print("Testing Stutter Detection Connector")
    print("=" * 60)
    
    # Initialize detector with correct parameters
    models_dir = "Model/models/copy"
    detection_threshold = 0.5  # Use training threshold
    
    detector = StutterDetector(
        models_dir=models_dir,
        detection_threshold=detection_threshold
    )
    
    # Test with a sample audio file
    audio_file = "Recordings/Prolongation.wav"  # Update this path as needed
    
    if not os.path.exists(audio_file):
        print(f"\n⚠️  Audio file not found: {audio_file}")
        print("Please update the audio_file path in test_connector.py")
        return
    
    # Process the audio file
    results = detector.process_audio_file(audio_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = detector.get_summary(results)
    for stutter_type, stats in summary.items():
        count = stats['count']
        pct = stats['percentage']
        print(f"{stutter_type:20s}: {count} detections ({pct:.1f}%)")
    
    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    test_connector()
