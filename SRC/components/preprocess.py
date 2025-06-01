# PPE Preprocessing & Box‑Level Undersampling Pipeline
# Refactored into reusable functions for filtering and undersampling

import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
VALID_CLASSES      = [1,6,2,7,3,8,5,9,13,10,11]
TARGET_CLASS_OLD   = 3    # old index for "Hardhat"
TARGET_CLASS_NEW   = VALID_CLASSES.index(TARGET_CLASS_OLD)
TARGET_COUNT       = 9000 # desired total hardhat boxes in training

# --- FUNCTIONS ---

def filter_and_reindex_split(
    src_img_dir: str,
    src_lbl_dir: str,
    dst_img_dir: str,
    dst_lbl_dir: str,
    valid_classes: list[int]
):
    """
    Filters and reindexes a dataset split:
      - Keeps only labels in valid_classes
      - Reindexes class IDs to 0..len(valid_classes)-1
      - Copies matching images
    """
    src_img = Path(src_img_dir)
    src_lbl = Path(src_lbl_dir)
    dst_img = Path(dst_img_dir); dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl = Path(dst_lbl_dir); dst_lbl.mkdir(parents=True, exist_ok=True)
    mapping = {old:new for new,old in enumerate(valid_classes)}

    for lbl_file in src_lbl.glob("*.txt"):
        new_lines = []
        for line in open(lbl_file):
            parts = line.strip().split()
            cls = int(parts[0])
            if cls in mapping:
                new_cls = mapping[cls]
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
        if not new_lines:
            continue
        # write filtered label
        (dst_lbl/lbl_file.name).write_text(''.join(new_lines))
        # copy image
        img_file = src_img / f"{lbl_file.stem}.jpg"
        if img_file.exists():
            shutil.copy(img_file, dst_img/img_file.name)


def undersample_hardhat_boxes(
    src_img_dir: str,
    src_lbl_dir: str,
    dst_img_dir: str,
    dst_lbl_dir: str,
    target_cls: int,
    target_count: int
):
    """
    From a filtered split, undersample hardhat instances to target_count boxes:
      - Collect all (filename, line) for target_cls
      - Randomly sample target_count of them
      - Rewrite each label file: keep all non-target_cls lines + sampled lines
      - Copy images with >=1 label to dst
    """
    src_img = Path(src_img_dir)
    src_lbl = Path(src_lbl_dir)
    dst_img = Path(dst_img_dir); dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl = Path(dst_lbl_dir); dst_lbl.mkdir(parents=True, exist_ok=True)

    # collect all hardhat instances
    all_instances = []
    for lbl in src_lbl.glob("*.txt"):
        for line in open(lbl):
            if int(line.split()[0]) == target_cls:
                all_instances.append((lbl.stem, line.rstrip()))

    keep_set = set(random.sample(all_instances, min(target_count, len(all_instances))))

    # rewrite labels
    for lbl in src_lbl.glob("*.txt"):
        stem = lbl.stem
        new_lines = []
        # add non-hardhat lines
        for line in open(lbl):
            if int(line.split()[0]) != target_cls:
                new_lines.append(line)
        # add kept hardhat lines for this file
        for (s, text) in keep_set:
            if s == stem:
                new_lines.append(text + "\n")
        if not new_lines:
            continue
        # write label & copy image
        (dst_lbl/f"{stem}.txt").write_text(''.join(new_lines))
        img_file = src_img/f"{stem}.jpg"
        if img_file.exists():
            shutil.copy(img_file, dst_img/img_file.name)


# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    # 1. Filter both splits
    filter_and_reindex_split(
        src_img_dir = 'data/train/images',
        src_lbl_dir = 'data/train/labels',
        dst_img_dir = 'data/train_filtered/images',
        dst_lbl_dir = 'data/train_filtered/labels',
        valid_classes=VALID_CLASSES
    )
    filter_and_reindex_split(
        src_img_dir = 'data/valid/images',
        src_lbl_dir = 'data/valid/labels',
        dst_img_dir = 'data/valid_filtered/images',
        dst_lbl_dir = 'data/valid_filtered/labels',
        valid_classes=VALID_CLASSES
    )

    # 2. Undersample hardhat **only on train_filtered → train_final**
    undersample_hardhat_boxes(
        src_img_dir   = 'data/train_filtered/images',
        src_lbl_dir   = 'data/train_filtered/labels',
        dst_img_dir   = 'data/train_final/images',
        dst_lbl_dir   = 'data/train_final/labels',
        target_cls    = TARGET_CLASS_NEW,
        target_count  = TARGET_COUNT
    )

    # 3. Leave valid **completely untouched** except for the initial filtering:
    #    copy filtered → final (no undersampling whatsoever)
    shutil.copytree('data/valid_filtered/images',
                    'data/valid_final/images',
                    dirs_exist_ok=True)
    shutil.copytree('data/valid_filtered/labels',
                    'data/valid_final/labels',
                    dirs_exist_ok=True)

    print('✔ Preprocessing complete: train_final & valid_final ready.')