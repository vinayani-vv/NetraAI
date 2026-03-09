# Violence Detection Dataset

Place your extracted video dataset here with the following structure:

```
datasets/
├── violence_dataset/
│   ├── violent_videos/     # Raw violent videos (.mp4, .avi)
│   ├── peaceful_videos/     # Raw peaceful videos (.mp4, .avi)
│   ├── images/
│   │   ├── train/           # Extracted training frames
│   │   └── val/             # Extracted validation frames
│   └── labels/
│       ├── train/           # YOLO format labels (.txt)
│       └── val/             # YOLO format labels (.txt)
```

## For Video Classification

If your dataset is organized as video clips for classification:
- violent/*.mp4 → Label 0
- peaceful/*.mp4 → Label 1

Use `extract_frames.py` to convert videos to training images.

## Frame Extraction

Run the frame extraction script to convert videos to frames:
```python
python extract_frames.py
```

## YOLO Format Labels

Each label file should contain:
- One object per line: `class_id center_x center_y width height`
- Coordinates normalized (0-1)
- Classes: 0=violent, 1=peaceful
