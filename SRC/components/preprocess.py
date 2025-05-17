# scripts/undersample_hardhat.py
import random
import shutil
from pathlib import Path

def undersample_hardhat(
    src_images="data/train/images",
    src_labels="data/train/labels",
    dst_images="data/train_small/images",
    dst_labels="data/train_small/labels",
    target_class=3,
    target_count=9000
):
    src_im = Path(src_images)
    src_lb = Path(src_labels)
    dst_im = Path(dst_images); dst_lb = Path(dst_labels)
    dst_im.mkdir(parents=True, exist_ok=True)
    dst_lb.mkdir(parents=True, exist_ok=True)

    # 1. Collect all hardhat files
    hardhat = []
    for label in src_lb.glob("*.txt"):
        with open(label) as f:
            for line in f:
                if line.split()[0] == str(target_class):
                    hardhat.append(label.stem)
                    break

    # 2. Sample target_count of them
    keep_hardhat = set(random.sample(hardhat, min(target_count, len(hardhat))))

    # 3. Copy undersampled train set
    for label in src_lb.glob("*.txt"):
        stem = label.stem
        # keep if not hardhat OR in sampled set
        if stem not in hardhat or stem in keep_hardhat:
            shutil.copy(src_lb/label.name, dst_lb/label.name)
            shutil.copy(src_im/f"{stem}.jpg", dst_im/f"{stem}.jpg")

    print(f"Created {len(list(dst_im.glob('*.jpg')))} images in {dst_images}")

if __name__ == "__main__":
    undersample_hardhat()