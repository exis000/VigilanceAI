import cv2
import os
from pathlib import Path

def validate_dataset(data_dir):
    """
    
    data_dir = ur data path so its "data/train" or you can also valid and test to check them too
    
    so if you wanna do a data check just change file path to what you want
    
    Validates images and labels 
    - Checks for corrupt images
    - Validates bounding boxes (0 <= x_center, y_center, width, height <= 1)
    - Checks for non-positive width/height
    - Ensures class IDs are within expected range
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    
    corrupt_images = []
    invalid_boxes = []
    invalid_classes = []

    # Iterate through all images
    for img_path in image_dir.glob("*.jpg"):
        # Check if image is readable
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Image is corrupt or unreadable")
            h, w, _ = img.shape
        except Exception as e:
            corrupt_images.append(str(img_path))
            continue

        # Check corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_boxes.append((str(label_path), line))
                continue

            class_id, x_center, y_center, bw, bh = map(float, parts)
            class_id = int(class_id)

            # Check class ID validity
            if not (0 <= class_id <= 13):  # Adjust max class ID as needed
                invalid_classes.append((str(label_path), class_id))

            # Check bounding box validity
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                invalid_boxes.append((str(label_path), line))
            if bw <= 0 or bh <= 0:
                invalid_boxes.append((str(label_path), line))

    # Print results
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Invalid boxes: {len(invalid_boxes)}")
    print(f"Invalid class IDs: {len(invalid_classes)}")

    return {
        "corrupt_images": corrupt_images,
        "invalid_boxes": invalid_boxes,
        "invalid_classes": invalid_classes
    }

# Usage
if __name__ == "__main__":
    data_dir = "data/processed/PPE_SAFETY_DATASET/valid"
    results = validate_dataset(data_dir)

"""output: no corrupt images , invalid boxes, invalid class id's
          are detected in both train and valid dataset   
          
          Corrupt images: 0
          Invalid boxes: 0
          Invalid class IDs: 0
"""