from ultralytics import YOLO
import shutil
from pathlib import Path

model = YOLO("runs/train/VigilanceAI_test9/weights/best.pt")
CLASS_THRESH = {0:0.4, 1:0.6, 2:0.6, 3:0.7, 4:0.6}  # person, gloves, glasses, vest, helmet

IMG_DIR    = Path("data/filtered_dataset_Copy/images/train")
GT_DIR     = Path("data/filtered_dataset_Copy/labels/train")
AUTO_DIR   = Path("data/filtered_dataset_Copy/auto_labels/train")
AUTO_DIR.mkdir(parents=True, exist_ok=True)


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


def xyxy_to_yolo(x1,y1,x2,y2,w,h):
    # convert pixel xyxy → normalized xywh
    xc = ((x1+x2)/2)/w;  yc = ((y1+y2)/2)/h
    bw = (x2-x1)/w;      bh = (y2-y1)/h
    return xc, yc, bw, bh

for img_path in IMG_DIR.glob("*.*"):
    img = str(img_path)
    stem = img_path.stem
    # load existing GT
    gt = []
    for line in open(GT_DIR/f"{stem}.txt"):
        c,x,y,w_,h_ = map(float, line.split())
        gt.append((int(c), (x,y,w_,h_)))
    # run inference with per‑class thresholds
    preds = model.predict(source=img, verbose=False)[0]
    new_lines = []
    for box,cls,conf in zip(preds.boxes.xyxy, preds.boxes.cls, preds.boxes.conf):
        c = int(cls)
        if conf < CLASS_THRESH[c]:
            continue
        x1,y1,x2,y2 = map(int, box)
        # check overlap with any GT of same class
        overlap = False
        for gc, (gx,gy,gw,gh) in gt:
            if gc!=c: continue
            # convert gt to pixels for IoU
            pw,ph = preds.orig_shape[1], preds.orig_shape[0]
            gx1,gy1,gx2,gy2 = xyxy_to_yolo(gx,gy,gw,gh,pw,ph)  # compute IoU accordingly
            # (compute IoU here…)
            if xyxy_iou((x1,y1,x2,y2),(gx1,gy1,gx2,gy2))>0.5:
                overlap=True; break
        if not overlap:
            xc,yc,bw,bh = xyxy_to_yolo(x1,y1,x2,y2,preds.orig_shape[1],preds.orig_shape[0])
            new_lines.append(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # write new pseudo labels
    if new_lines:
        with open(AUTO_DIR/f"{stem}.txt","w") as f:
            f.write("\n".join(new_lines))