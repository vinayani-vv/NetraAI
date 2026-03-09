"""Check and fix mislabeled images in the dataset"""

import os
from pathlib import Path

def list_images_with_context():
    """List all images in source folders"""
    base = Path('datasets/violence_dataset')
    
    violent = list((base / 'violence').glob('*.jpg'))
    peaceful = list((base / 'non_violence').glob('*.jpg'))
    
    print("=" * 60)
    print("VIOLENCE FOLDER IMAGES (should be violent):")
    print("=" * 60)
    for img in sorted(violent):
        print(f"  {img.name}")
    
    print("\n" + "=" * 60)
    print("NON_VIOLENCE FOLDER IMAGES (should be peaceful):")
    print("=" * 60)
    for img in sorted(peaceful):
        print(f"  {img.name}")
    
    print(f"\nTotal violent images: {len(violent)}")
    print(f"Total peaceful images: {len(peaceful)}")

def check_label_distribution():
    """Check label distribution in train/val sets"""
    base = Path('datasets/violence_dataset')
    
    train_labels = list((base / 'labels' / 'train').glob('*.txt'))
    val_labels = list((base / 'labels' / 'val').glob('*.txt'))
    
    train_class_0 = 0
    train_class_1 = 0
    val_class_0 = 0
    val_class_1 = 0
    
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('0 '):
                train_class_0 += 1
            elif content.startswith('1 '):
                train_class_1 += 1
    
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('0 '):
                val_class_0 += 1
            elif content.startswith('1 '):
                val_class_1 += 1
    
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION (train + val):")
    print("=" * 60)
    print(f"Train class 0 (violent): {train_class_0}")
    print(f"Train class 1 (peaceful): {train_class_1}")
    print(f"Val class 0 (violent): {val_class_0}")
    print(f"Val class 1 (peaceful): {val_class_1}")
    print(f"\nExpected violent (from violence folder): {len(list((base / 'violence').glob('*.jpg')))}")
    print(f"Expected peaceful (from non_violence folder): {len(list((base / 'non_violence').glob('*.jpg')))}")

def find_duplicate_names():
    """Check for duplicate image names across folders"""
    base = Path('datasets/violence_dataset')
    
    violent_names = set((base / 'violence').glob('*.jpg'))
    peaceful_names = set((base / 'non_violence').glob('*.jpg'))
    
    duplicates = violent_names.intersection(peaceful_names)
    
    if duplicates:
        print("\n" + "=" * 60)
        print("WARNING: DUPLICATE IMAGE NAMES FOUND!")
        print("=" * 60)
        for d in sorted(duplicates):
            print(f"  {d.name} exists in both folders!")

def suggest_fix():
    """Suggest how to fix the dataset"""
    print("\n" + "=" * 60)
    print("RECOMMENDED FIX:")
    print("=" * 60)
    print("""
The issue is likely that some images in the 'violence/' and 'non_violence/'
folders are actually miscategorized. 

To fix:
1. Manually check images in each folder
2. Move miscategorized images to the correct folder
3. Re-run setup_dataset.py to regenerate the train/val splits

Suggested action:
- Go through 'datasets/violence_dataset/violence/' folder
- Go through 'datasets/violence_dataset/non_violence/' folder  
- Verify each image is in the correct category
- Then re-run: python setup_dataset.py
""")

if __name__ == "__main__":
    list_images_with_context()
    check_label_distribution()
    find_duplicate_names()
    suggest_fix()
