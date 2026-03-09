"""
Violence Detection Model Training Script
Using YOLOv8 for image dataset training
"""

from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import random

def setup_dataset_structure():
    """Set up YOLO dataset structure from existing folders"""
    
    base_path = Path('datasets/violence_dataset')
    
    # Find image folders
    violent_folder = base_path / 'violence'
    peaceful_folder = base_path / 'non_violence'
    
    # Check if folders exist
    if not violent_folder.exists():
        print(f"Error: {violent_folder} not found!")
        return False
    if not peaceful_folder.exists():
        print(f"Error: {peaceful_folder} not found!")
        return False
    
    # Create YOLO structure
    images_train = base_path / 'images' / 'train'
    images_val = base_path / 'images' / 'val'
    labels_train = base_path / 'labels' / 'train'
    labels_val = base_path / 'labels' / 'val'
    
    for folder in [images_train, images_val, labels_train, labels_val]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    violent_images = list(violent_folder.glob('*.jpg')) + list(violent_folder.glob('*.jpeg')) + list(violent_folder.glob('*.png'))
    peaceful_images = list(peaceful_folder.glob('*.jpg')) + list(peaceful_folder.glob('*.jpeg')) + list(peaceful_folder.glob('*.png'))
    
    print(f"Found {len(violent_images)} violent images")
    print(f"Found {len(peaceful_images)} peaceful images")
    
    # Split: 80% train, 20% val
    def split_and_copy(images, label_class, class_name):
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img_path in train_images:
            # Use unique filename to prevent overwrites
            new_name = f"{class_name}_{img_path.name}"
            shutil.copy(img_path, images_train / new_name)
            create_label(labels_train / Path(new_name).with_suffix('.txt').name, label_class)
        
        for img_path in val_images:
            # Use unique filename to prevent overwrites
            new_name = f"{class_name}_{img_path.name}"
            shutil.copy(img_path, images_val / new_name)
            create_label(labels_val / Path(new_name).with_suffix('.txt').name, label_class)
        
        print(f"  Copied {len(train_images)} to train, {len(val_images)} to val")
    
    # Process both classes
    print("Processing violent images...")
    split_and_copy(violent_images, label_class=0, class_name='violent')
    
    print("Processing peaceful images...")
    split_and_copy(peaceful_images, label_class=1, class_name='peaceful')
    
    print("Dataset structure ready!")
    return True

def create_label(label_path, class_id):
    """Create a YOLO format label file with full-frame bounding box"""
    # Full frame bounding box: class_id center_x center_y width height
    # All values normalized (0-1)
    with open(label_path, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

def create_data_yaml():
    """Create data.yaml for YOLOv8 training"""
    
    data_yaml = """path: ./datasets/violence_dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: violent
  1: peaceful
"""
    
    with open('datasets/violence_dataset/data.yaml', 'w') as f:
        f.write(data_yaml)
    
    print("data.yaml created!")

def train_violence_model(epochs=100, imgsz=640):
    """Train YOLOv8 on violence detection dataset"""
    
    # Setup dataset structure
    if not setup_dataset_structure():
        return
    
    # Create data.yaml
    create_data_yaml()
    
    # Load YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data='datasets/violence_dataset/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name='violence_detection',
        workers=0,  # Windows compatibility
        val=True    # Validate during training
    )
    
    # Export to ONNX format
    print("\nExporting model to ONNX...")
    model.export(format='onnx')
    
    print(f"\n✅ Training complete!")
    print(f"Model saved to: runs/detect/violence_detection/")
    print(f"Best model: runs/detect/violence_detection/weights/best.pt")
    
    return results

def retrain_with_custom_model():
    """Use the trained model for inference"""
    
    model_path = 'runs/detect/violence_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("Model not found! Please train first.")
        return
    
    model = YOLO(model_path)
    
    # Test on an image
    results = model.predict(
        source='static/uploads/',
        save=True,
        conf=0.5
    )
    
    print("Inference complete!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Violence Detection Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--retrain', action='store_true', help='Run inference with trained model')
    
    args = parser.parse_args()
    
    if args.retrain:
        retrain_with_custom_model()
    else:
        train_violence_model(epochs=args.epochs, imgsz=args.imgsz)
