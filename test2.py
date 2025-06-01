import cv2
import os
from pathlib import Path

# === CONFIGURATION ===
image_dir = Path("data/train/images")
label_dir = Path("data/train/labels")
show_limit = 200  # how many images to preview
img_ext = ".jpg"  # or ".png"

# === MAIN LOOP ===
image_files = list(image_dir.glob(f"*{img_ext}"))

for idx, img_path in enumerate(image_files):
    if idx >= show_limit:
        break

    label_path = label_dir / (img_path.stem + ".txt")

    if not label_path.exists():
        continue  # skip if no label file

    # Load image
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    # Read YOLO label
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            # Convert YOLO (normalized) to pixel coords
            xc, yc = x_center * w, y_center * h
            bw, bh = box_w * w, box_h * h
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID {class_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show image
    cv2.imshow("YOLO Reverse Label Viewer", img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()


"""
    Class 0: safety_vest_absent
    Class 1: gloves_present
    Class 2: glasses_present
    Class 3: helmet
    Class 4: stairs
    Class 5: facemask
    Class 6: gloves_absent
    Class 7: glasses_absent
    Class 8: human
    Class 9: no face mask
    Class 10: formal suit?
    Class 11: safety_vest_present
    Class 12: cones
    Class 13: reflector suit
    
    
path: ../data
train: train_final/images
val: valid_final/images
nc: 11
names:
  0: gloves
  1: no_gloves
  2: safety_glasses
  3: no_safety_glasses
  4: hardhat
  5: no_hardhat
  6: mask
  7: no_mask
  8: safety_vest
  9: no_safety_vest
  10: person


file structure

vigilance-ai/  
├── data/  
│   ├── raw/                  # (Git‑ignored) original downloads  
│   ├── processed/            # cleaned + split images & labels  
│   │   ├── train/            # YOLOv11 “images” & “labels” subfolders  
│   │   └── val/  
│   └── pseudo/               # auto‑labels from model (for self‑training)  
├── src/  
│   ├── __init__.py  
│   ├── config.py             # hyperparams, paths, YAML loader  
│   ├── data/  
│   │   ├── check_data.py     # integrity checks, distribution reports  
│   │   └── preprocess.py     # letterbox, normalization, anchor k‑means  
│   ├── train.py              # model training entry point  
│   ├── infer.py              # CLI for batch inference + pseudo‑labeling  
│   ├── metrics.py            # mAP, precision/recall calculators  
│   ├── visualize.py          # augmentation preview, result plotting  
│   ├── api/  
│   │   ├── server.py         # FastAPI or Flask service wrapping model  
│   │   └── requirements.txt  # minimal dependencies for serving  
│   └── deploy/  
│       ├── Dockerfile        # containerize API + model weights  
│       └── k8s.yaml          # (if using Kubernetes)  
├── notebooks/                # EDA, augmentation display, proof‑of‑concepts  
│   └── 01_data_checks.ipynb  
├── runs/                     # training logs, TensorBoard, model checkpoints  
├── requirements.txt          # pin exact versions (torch, ultralytics, cv2…)  
├── environment.yml           # conda environment spec (if desired)  
├── README.md                 # project overview + quickstart  
├── CHANGELOG.md  
├── .gitignore  
└── .github/                  # CI workflows, issue templates, PR templates  
    └── workflows/  
        └── ci.yml           # run lint, tests, small smoke‑train on PR  


2nd version
===========================================================================================================



vigilance‑ai/                 ← Root of your Git repo
├── data/                     ← Raw and processed data (git‑ignored)
│   ├── raw/                  ← Original downloads (S3, cameras, etc.)
│   ├── processed/            ← After cleaning, splitting (train/val), augmentation
│   │   ├── train/
│   │   └── val/
│   └── pseudo/               ← Pseudo‑labels from inference
│
├── models/                   ← Model weights & checkpoints
│   ├── yolov11x.pt           ← Base pretrained weights (checked in or downloaded via script)
│   ├── best.pt               ← Your best fine‑tuned checkpoint
│   └── last.pt               ← Last epoch checkpoint
│
├── src/                      ← Core Python source
│   ├── __init__.py
│   ├── config.py             ← All paths, hyperparams, data.yaml loader
│   │
│   ├── data/                 ← Data utilities
│   │   ├── check_data.py     ← Integrity & distribution checks
│   │   ├── preprocess.py     ← Letterbox, normalize, anchor‑kmeans
│   │   └── make_yaml.py      ← Script to auto‑generate data.yaml
│   │
│   ├── train.py              ← Training entry point (calls ultralytics.train)
│   ├── infer.py              ← Batch inference & pseudo‑labeling CLI
│   ├── metrics.py            ← mAP / precision‐recall calculators
│   ├── visualize.py          ← Augmentation previews, result plotting
│   │
│   ├── api/                  ← Model‑serving code
│   │   ├── server.py         ← FastAPI or Flask inference endpoints
│   │   ├── requirements.txt  ← Minimal deps just for serving
│   │   └── Dockerfile        ← Containerizing the API
│   │
│   └── utils/                ← Miscellaneous scripts
│       ├── download_data.sh  ← Bash to fetch raw data
│       └── convert_annotations.py
│
├── webapp/                   ← Frontend or dashboard
│   ├── package.json          ← If JS (React/Vue) or `requirements.txt` for Flask/Streamlit
│   ├── public/               ← Static assets
│   ├── src/                  ← React/Vue or Flask templates
│   └── Dockerfile            ← Containerize the UI
│
├── deployment/               ← Orchestration & infra configs
│   ├── k8s.yaml              ← Kubernetes manifests (Deployments, Services)
│   ├── helm/                 ← Helm charts
│   └── nginx/                ← Reverse‑proxy configs
│
├── tests/                    ← Unit + integration tests
│   ├── unit/                 ← pytest for data checks, metrics, small modules
│   └── integration/          ← Smoke train/infer tests on CI
│
├── notebooks/                ← EDA, POC, error analysis
│   ├── 01_data_checks.ipynb
│   └── 02_augmentation.ipynb
│
├── docs/                     ← High‑level docs & architecture
│   ├── architecture.md
│   ├── user_guide.md
│   └── api_reference.md
│
├── runs/                     ← Training logs, TensorBoard, W&B outputs (git‑ignored)
│
├── .github/                  ← GitHub Actions workflows, issue/PR templates
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── requirements.txt          ← Pinned Python deps for training & dev
├── environment.yml           ← Conda env spec (if used)
├── .gitignore
├── README.md                 ← Project overview & quickstart
└── CHANGELOG.md              ← Versioned changes

"""