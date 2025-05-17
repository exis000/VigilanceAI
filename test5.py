# from ultralytics import YOLO
# import cv2
# from pathlib import Path
# import random
# model = YOLO("yolo11x.pt")

# # Path to your image directory
# image_dir = Path("data/test/images")

# # Get all image paths (adjust the pattern to match your file types)
# image_paths = list(image_dir.glob("*.jpg"))  # use "*.png" or "*" to match all images if needed

# # Randomly sample 20 images
# subset_paths = random.sample(image_paths, 20)

# # Run inference on the selected images

# results = model.predict(source=subset_paths, device="cpu", save=True)
  
  
"""
1)save=True
This saves images with bounding boxes drawn on them (as .jpg or .png) to the output folder.
2) save_txt=True
This saves the prediction results in YOLO label format (.txt) for each image.

Each .txt file contains: <class_id> <x_center> <y_center> <width> <height> <confidence>

save_conf=True
This adds confidence scores to the .txt files (only if save_txt=True is also set).

Format with confidence: 0 0.516 0.334 0.123 0.088 0.92  # <- last number is confidence

project= and name=
These control where the results are saved:


project="runs/detect"	Top-level folder

name="ppe_sample"	Subfolder name inside project
   
 So it saves to:
runs/detect/ppe_sample/

default is just runs/detect/predict  
    """ 
    
    
# Print results
# for r in results:
#     print(r)
# scripts/check_distribution.py
from pathlib import Path
from collections import Counter, defaultdict

# def count_labels(label_dir="data/train_small/labels"):
#     label_dir = Path(label_dir)
#     # Counter for total bounding‐box instances per class
#     class_counts = Counter()
#     # Set to track which images contain which classes
#     images_per_class = defaultdict(set)

#     for label_file in label_dir.glob("*.txt"):
#         stem = label_file.stem
#         with open(label_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 cls = int(parts[0])
#                 class_counts[cls] += 1
#                 images_per_class[cls].add(stem)

#     # Print results
#     print("Class ID | #Boxes | #Images")
#     print("----------------------------")
#     for cls in sorted(class_counts):
#         print(f"{cls:>7} | {class_counts[cls]:>6} | {len(images_per_class[cls]):>7}")

# if __name__ == "__main__":
#     count_labels()

from ultralytics import YOLO
model = YOLO("runs/train/VigilanceAI_V1/weights/best.pt")
results = model.predict(source="data/test/images", max_det=100, save=True)

for i in results[0]:
    print (i)