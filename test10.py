from pathlib import Path
import shutil

# Map new dataset class indices to your unified class indices (from your YAML file)
CLASS_MAPPING = {
    0: 2,   # Glass       → goggles
    1: 0,   # Gloves      → gloves
    2: 4,   # Helmet      → hardhat
    3: 5,   # No-Helmet   → no_hardhat
    4: 9,   # No-Vest     → no_safety_vest
    5: 10,  # Person      → person
    6: 8    # Safety-Vest → safety_vest
}

def clean_and_remap_labels(
    src_lbl_dir: str,
    dst_lbl_dir: str,
    src_img_dir: str,
    dst_img_dir: str,
    img_ext: str = ".jpg"
):
    src_lbl = Path(src_lbl_dir)
    dst_lbl = Path(dst_lbl_dir)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    src_img = Path(src_img_dir)
    dst_img = Path(dst_img_dir)
    dst_img.mkdir(parents=True, exist_ok=True)

    for lbl_file in src_lbl.glob("*.txt"):
        lines = lbl_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            old_cls = int(parts[0])
            if old_cls in CLASS_MAPPING:
                parts[0] = str(CLASS_MAPPING[old_cls])
                new_lines.append(" ".join(parts))

        if not new_lines:
            # Skip images with no valid labels
            img_path = src_img / (lbl_file.stem + img_ext)
            if img_path.exists():
                img_path.unlink(missing_ok=True)
            continue

        # Save remapped label
        (dst_lbl / lbl_file.name).write_text("\n".join(new_lines) + "\n")

        # Copy image
        img_path = src_img / (lbl_file.stem + img_ext)
        if img_path.exists():
            shutil.copy(img_path, dst_img / img_path.name)

    print(f"✅ Remapping complete. Cleaned data saved to {dst_img.parent}")

# ───────────── USAGE ─────────────
if __name__ == "__main__":
    clean_and_remap_labels(
        src_lbl_dir="data/new_data_for_general_safetyvest/labels",
        dst_lbl_dir="data/new_data_general_filtered/labels",
        src_img_dir="data/new_data_for_general_safetyvest/images",
        dst_img_dir="data/new_data_general_filtered/images",
        img_ext=".jpg"  # change to ".png" if needed
    )