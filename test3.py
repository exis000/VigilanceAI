

from pathlib import Path

def check_split(data_dir):
    splits = ['train', 'valid']
    for split in splits:
        img_dir = Path(data_dir) / split / 'images'
        label_dir = Path(data_dir) / split / 'labels'
        assert img_dir.exists(), f"Missing {split}/images"
        assert label_dir.exists(), f"Missing {split}/labels"
        print(f"{split}: {len(list(img_dir.glob('*.jpg')))} images, {len(list(label_dir.glob('*.txt')))} labels")

check_split("data")