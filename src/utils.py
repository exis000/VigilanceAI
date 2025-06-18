

    
import os
import time
from huggingface_hub import HfApi, HfFolder

# ───────── CONFIG ─────────

# Make sure HF_TOKEN is set in your environment
token = ("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")
if not token:
    raise RuntimeError("Please set HF_TOKEN environment variable with write access")

REPO_ID      = "exis0705/PPE-DATASET"   # your dataset repo
BASE_FOLDER  = "data/combined_dataset"  # local path to your merged dataset
BATCH_SIZE   = 500                      # number of files to upload per sub‐batch
PAUSE_SEC    = 1.0                      # pause between batches to avoid rate‐limit

# ───────── SCRIPT ─────────

api = HfApi()

def upload_split(split_path, path_in_repo_prefix):
    """Uploads all files under split_path in batches."""
    files = sorted(os.listdir(split_path))
    total = len(files)
    print(f"\nUploading {total} files from {split_path} → {path_in_repo_prefix}/ ...")
    for i in range(0, total, BATCH_SIZE):
        batch = files[i : i + BATCH_SIZE]
        for fname in batch:
            local_path = os.path.join(split_path, fname)
            # remote path: e.g. "combined_dataset/images/train/img123.jpg"
            remote_path = f"{path_in_repo_prefix}/{fname}"
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                token=token,
                repo_type="dataset"
            )
        print(f"  • Uploaded files {i+1}–{min(i+BATCH_SIZE, total)} / {total}")
        time.sleep(PAUSE_SEC)

def main():
    splits = [
        ("images/train", "combined_dataset/images/train"),
        ("images/val",   "combined_dataset/images/val"),
        ("labels/train", "combined_dataset/labels/train"),
        ("labels/val",   "combined_dataset/labels/val"),
    ]
    for local_suffix, remote_prefix in splits:
        local_path = os.path.join(BASE_FOLDER, local_suffix)
        upload_split(local_path, remote_prefix)
    print("\n✅ All batches uploaded!")


def xyxy_iou(box1, box2):
    """
    Compute IoU between two boxes: each box = (x1,y1,x2,y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def ex():
    print("this wokrs tho")


if __name__ == "__main__":
    main()