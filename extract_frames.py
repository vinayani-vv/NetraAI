"""
Extract frames from violence detection videos for training
"""

import cv2
import os
from pathlib import Path

def extract_frames_from_videos(video_dir, output_dir, label, frames_per_video=10):
    """Extract frames from videos in a directory"""
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    supported_formats = ['.mp4', '.avi', '.mkv', '.mov']
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in supported_formats]
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    frame_count = 0
    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // frames_per_video)
        
        current_frame = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % frame_interval == 0:
                # Save frame
                frame_path = output_dir / f"{video_path.stem}_frame{saved_count}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Create label file
                label_path = output_dir / f"{video_path.stem}_frame{saved_count}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"{label} 0.5 0.5 0.5 0.5\n")  # Full frame bounding box
                
                saved_count += 1
                frame_count += 1
            
            current_frame += 1
        
        cap.release()
        print(f"Processed {video_path.name}: {saved_count} frames extracted")
    
    return frame_count

def prepare_dataset():
    """Prepare dataset from video folders"""
    
    base_path = Path('datasets/violence_dataset')
    
    # Extract violent frames (label=0)
    violent_videos = base_path / 'violent_videos'
    violent_frames = base_path / 'images' / 'train'
    if violent_videos.exists():
        extract_frames_from_videos(violent_videos, violent_frames, label=0)
    
    # Extract peaceful frames (label=1)
    peaceful_videos = base_path / 'peaceful_videos'
    peaceful_frames = base_path / 'images' / 'train'
    if peaceful_videos.exists():
        extract_frames_from_videos(peaceful_videos, peaceful_frames, label=1)
    
    print("Frame extraction complete!")

if __name__ == '__main__':
    prepare_dataset()
