from pathlib import Path

def delete_images_with_no_labels(
    img_dir: str,
    lbl_dir: str,
    img_ext: str = ".jpg",
    delete_empty_labels: bool = True
):
    """
    Deletes images that have no labels (missing or empty .txt files).
    
    Args:
      img_dir: Path to images folder
      lbl_dir: Path to labels folder
      img_ext: File extension for images (e.g., '.jpg', '.png')
      delete_empty_labels: Whether to also delete the empty label files
    """
    img_path = Path(img_dir)
    lbl_path = Path(lbl_dir)
    removed_count = 0

    for img_file in img_path.glob(f"*{img_ext}"):
        lbl_file = lbl_path / f"{img_file.stem}.txt"
        
        # If label doesn't exist or is empty
        if not lbl_file.exists() or lbl_file.read_text().strip() == "":
            img_file.unlink()  # Delete image
            if delete_empty_labels and lbl_file.exists():
                lbl_file.unlink()  # Delete empty label
            removed_count += 1

    print(f"✅ Deleted {removed_count} images with no labels.")

# ───────────── USAGE ─────────────
if __name__ == "__main__":
    delete_images_with_no_labels(
        img_dir="data/valid_final/images",
        lbl_dir="data/valid_final/labels",
        img_ext=".jpg",
        delete_empty_labels=True  # Also delete .txt files if they're empty
    )