"""Fix the dataset by using unique filenames to prevent overwrites"""

import os
import shutil
from pathlib import Path

def fix_dataset():
    base = Path('datasets/violence_dataset')
    
    # Clean up existing train/val folders
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        folder_path = base / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
    
    # Get images from source folders with unique prefixes
    violent = list((base / 'violence').glob('*.jpg'))
    peaceful = list((base / 'non_violence').glob('*.jpg'))
    
    print(f"Violent images: {len(violent)}")
    print(f"Peaceful images: {len(peaceful)}")
    
    # Create labeled lists with unique identifiers
    violent_labeled = [(img, 'violent', 0) for img in violent]
    peaceful_labeled = [(img, 'peaceful', 1) for img in peaceful]
    
    # Split 80/20 for each class separately
    def split_data(images, class_name, class_id):
        import random
        random.shuffle(images)
        split = int(len(images) * 0.8)
        train = images[:split]
        val = images[split:]
        
        for img, orig_name, cid in train:
            # Use unique filename: violent_0.jpg, peaceful_0.jpg
            new_name = f"{class_name}_{img.name}"
            shutil.copy(img, base / 'images' / 'train' / new_name)
            with open(base / 'labels' / 'train' / f"{Path(new_name).stem}.txt", 'w') as f:
                f.write(f"{cid} 0.5 0.5 1.0 1.0")
        
        for img, orig_name, cid in val:
            new_name = f"{class_name}_{img.name}"
            shutil.copy(img, base / 'images' / 'val' / new_name)
            with open(base / 'labels' / 'val' / f"{Path(new_name).stem}.txt", 'w') as f:
                f.write(f"{cid} 0.5 0.5 1.0 1.0")
        
        print(f"  {class_name.capitalize()}: Train={len(train)}, Val={len(val)}")
        return len(train), len(val)
    
    print("\nCopying with unique filenames...")
    vt, vv = split_data(violent_labeled, 'violent', 0)
    pt, pv = split_data(peaceful_labeled, 'peaceful', 1)
    
    print(f"\n{'='*50}")
    print("FIXED DATASET SUMMARY:")
    print(f"{'='*50}")
    print(f"Train: {vt + pt} images ({vt} violent + {pt} peaceful)")
    print(f"Val: {vv + pv} images ({vv} violent + {pv} peaceful)")
    print(f"Total: {vt + vv + pt + pv} images")
    
    print(f"\nTrain/Val split ratio: {((vt+pt)/(vt+pt+vv+pv))*100:.1f}% / {((vv+pv)/(vt+pt+vv+pv))*100:.1f}%")

if __name__ == "__main__":
    fix_dataset()
    print("\nDone! Re-run train_violence_model.py to train with corrected dataset.")
