from pathlib import Path
import shutil

def remap_class_in_labels(
    old_cls: int,
    new_cls: int,
    src_lbl_dir: str,
    dst_lbl_dir: str,
    src_img_dir: str = None,
    dst_img_dir: str = None,
    img_ext: str = ".jpg"
):
    """
    Remap every annotation line whose class==old_cls to new_cls.
    Writes fixed labels into dst_lbl_dir (same filenames).
    If src_img_dir & dst_img_dir are provided, also copies matching images.

    Args:
      old_cls:      the class ID to look for (e.g. 0)
      new_cls:      the class ID to replace it with (e.g. 4)
      src_lbl_dir:  path to folder of raw labels
      dst_lbl_dir:  path to write cleaned labels
      src_img_dir:  (optional) raw images folder
      dst_img_dir:  (optional) where to copy images
      img_ext:      image file extension (".jpg", ".png", etc.)
    """
    src_lbl = Path(src_lbl_dir)
    dst_lbl = Path(dst_lbl_dir); dst_lbl.mkdir(parents=True, exist_ok=True)

    # Optional image copy
    copy_images = False
    if src_img_dir and dst_img_dir:
        src_img = Path(src_img_dir)
        dst_img = Path(dst_img_dir); dst_img.mkdir(parents=True, exist_ok=True)
        copy_images = True

    for lbl_file in src_lbl.glob("*.txt"):
        lines = lbl_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            if cls == old_cls:
                parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))
        if not new_lines:
            continue  # no valid lines

        # write remapped label
        (dst_lbl / lbl_file.name).write_text("\n".join(new_lines) + "\n")

        # optionally copy the image only if paths differ
        if copy_images:
            img_name      = lbl_file.stem + img_ext
            src_img_path  = src_img / img_name
            dst_img_path  = dst_img / img_name
            if src_img_path.exists() and src_img_path != dst_img_path:
                shutil.copy(src_img_path, dst_img_path)

    print(f"Remapped class {old_cls}→{new_cls} in {dst_lbl_dir}"
          + (f" and copied images to {dst_img_dir}" if copy_images else ""))


# ──────────── USAGE EXAMPLE ────────────
if __name__ == "__main__":
    # Remap 0→4 in labels, but skip copying if images live in the same folder:
    remap_class_in_labels(
        old_cls      = 0,
        new_cls      = 4,
        src_lbl_dir  = "data/hardhats_new_data/labels",
        dst_lbl_dir  = "data/hardhats_new_data/labels",  # same as src
        src_img_dir  = "data/hardhats_new_data/images",
        dst_img_dir  = "data/hardhats_new_data/images",  # same as src
        img_ext      = ".jpg"
    )