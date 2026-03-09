"""Setup dataset structure for YOLOv8 training"""

import os
import shutil
from pathlib import Path

base = Path('datasets/violence_dataset')

# Create folders
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    (base / folder).mkdir(exist_ok=True)

# Get images
violent = list((base / 'violence').glob('*.jpg'))
peaceful = list((base / 'non_violence').glob('*.jpg'))

print(f"Violent: {len(violent)} images")
print(f"Peaceful: {len(peaceful)} images")

# Split and copy (80/20)
def split_copy(images, label):
    import random
    random.shuffle(images)
    split = int(len(images) * 0.8)
    train = images[:split]
    val = images[split:]
    
    for i, img in enumerate(train):
        shutil.copy(img, base / 'images' / 'train' / img.name)
        with open(base / 'labels' / 'train' / f"{img.stem}.txt", 'w') as f:
            f.write(f"{label} 0.5 0.5 1.0 1.0")
    
    for i, img in enumerate(val):
        shutil.copy(img, base / 'images' / 'val' / img.name)
        with open(base / 'labels' / 'val' / f"{img.stem}.txt", 'w') as f:
            f.write(f"{label} 0.5 0.5 1.0 1.0")
    
    print(f"  Train: {len(train)}, Val: {len(val)}")

split_copy(violent, 0)
split_copy(peaceful, 1)

print("Done!")
