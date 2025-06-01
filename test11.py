#!/usr/bin/env python3
import shutil
from pathlib import Path

# ─── CONFIGURATION ─────────────────────────────────────────────────────
TARGET_CLASS_ID = 0    # old ID for "hardhat"
NEW_CLASS_ID    = 0    # remapped ID in this 1-class dataset

# Source folders
TRAIN_IMG_DIR = Path("data/train_filtered/images")
TRAIN_LBL_DIR = Path("data/train_filtered/labels")
VAL_IMG_DIR   = Path("data/valid_filtered/images")
VAL_LBL_DIR   = Path("data/valid_filtered/labels")

# Destination folders
OUT_TRAIN_IMG = Path("data/Gloves_dataset/train/images")
OUT_TRAIN_LBL = Path("data/Gloves_dataset/train/labels")
OUT_VAL_IMG   = Path("data/Gloves_dataset/valid/images")
OUT_VAL_LBL   = Path("data/Gloves_dataset/valid/labels")

for d in (OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG,OUT_VAL_LBL):
    d.mkdir(parents=True, exist_ok=True)

# ─── FUNCTION TO FILTER + REMAP ─────────────────────────────────────────
def filter_and_remap(src_lbl: Path, src_img: Path, dst_lbl: Path, dst_img: Path,
                     old_id: int, new_id: int):
    for lbl in src_lbl.glob("*.txt"):
        lines = lbl.read_text().splitlines()
        out_lines = []
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            if cls == old_id:
                # rewrite class index to new_id
                out_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
        if not out_lines:
            continue
        # save remapped labels
        (dst_lbl / lbl.name).write_text("".join(out_lines))
        # copy image
        for ext in (".jpg", ".png", ".jpeg"):
            img = src_img / (lbl.stem + ext)
            if img.exists():
                shutil.copy(img, dst_img / img.name)
                break

# ─── RUN FOR TRAIN + VAL ─────────────────────────────────────────────────
print("Filtering+remapping TRAIN …")
filter_and_remap(TRAIN_LBL_DIR, TRAIN_IMG_DIR, OUT_TRAIN_LBL, OUT_TRAIN_IMG,
                 TARGET_CLASS_ID, NEW_CLASS_ID)

print("Filtering+remapping VAL …")
filter_and_remap(VAL_LBL_DIR,   VAL_IMG_DIR,   OUT_VAL_LBL,   OUT_VAL_IMG,
                 TARGET_CLASS_ID, NEW_CLASS_ID)

print("Done.")
print(f"➡️ Train: {len(list(OUT_TRAIN_IMG.glob('*')))} images, "
      f"{len(list(OUT_TRAIN_LBL.glob('*')))} labels")
print(f"➡️ Val:   {len(list(OUT_VAL_IMG.glob('*')))} images, "
      f"{len(list(OUT_VAL_LBL.glob('*')))} labels")