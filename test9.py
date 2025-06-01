import cv2
from pathlib import Path

def check_null_images(img_dir: str, img_ext: str = ".jpg"):
    """
    Checks for null or unreadable images in a directory.

    Args:
        img_dir: Path to the directory containing images.
        img_ext: File extension of images (e.g., '.jpg', '.png').

    Returns:
        List of corrupted or unreadable image file paths.
    """
    img_path = Path(img_dir)
    bad_images = []

    for img_file in img_path.glob(f"*{img_ext}"):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                bad_images.append(img_file)
        except Exception as e:
            print(f"⚠️ Error reading {img_file}: {e}")
            bad_images.append(img_file)

    if bad_images:
        print(f"\n❌ Found {len(bad_images)} unreadable/null images:")
        for file in bad_images:
            print(f" - {file}")
    else:
        print("✅ All images are readable.")

    return bad_images

# ───────────── USAGE ─────────────
if __name__ == "__main__":
    bad_files = check_null_images(img_dir="data/new_data_general_filtered/images", img_ext=".jpg")