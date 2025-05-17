import cv2
from pathlib import Path

# Configuration
DATA_DIR = Path("data/train")  # Adjusted to point to 'train'
CLASSES_TO_INSPECT = list(range(14))  # Classes 0–13
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # Bounding box colors

def yolo_to_pixels(yolo_coords, img_width, img_height):
    """Convert YOLO format to pixel coordinates."""
    x_center, y_center, w, h = yolo_coords
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    return (
        int(x_center - w / 2),
        int(y_center - h / 2),
        int(x_center + w / 2),
        int(y_center + h / 2)
    )

def find_class_samples(label_dir, image_dir, target_classes):
    """Find one image per class containing that class."""
    class_samples = {}
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id in target_classes and class_id not in class_samples:
                    image_path = image_dir / f"{label_file.stem}.jpg"
                    if image_path.exists():
                        class_samples[class_id] = image_path
                        if len(class_samples) == len(target_classes):
                            return class_samples
    return class_samples

# Paths
label_dir = DATA_DIR / "labels"
image_dir = DATA_DIR / "images"

# Find one example per class
class_samples = find_class_samples(label_dir, image_dir, CLASSES_TO_INSPECT)

# Display each class example
for class_id, image_path in class_samples.items():
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        continue

    img_h, img_w = img.shape[:2]
    label_path = label_dir / f"{image_path.stem}.txt"

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        label_class = int(parts[0])
        coords = list(map(float, parts[1:5]))
        x_min, y_min, x_max, y_max = yolo_to_pixels(coords, img_w, img_h)

        color = COLORS[class_id % len(COLORS)] if label_class == class_id else (200, 200, 200)
        thickness = 2 if label_class == class_id else 1

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(img, f"Class {label_class}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow(f"Class {class_id} - Press any key", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Inspection complete! Update your classes in data.yaml:")
print("names:")
for class_id in sorted(class_samples.keys()):
    print(f"  {class_id}: your_class_name  # Replace based on what you saw")