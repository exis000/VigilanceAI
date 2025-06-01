
from ultralytics import YOLO

def main():
    # Start from the official YOLOv11s pretrained weights
    model = YOLO("models/yolo11s.pt")

    model.train(
        data="data/Gloves.yaml",  
        epochs=100,             
        batch=16,
        imgsz=640,
        device="cuda",
    

        # Optimizer & LR schedule
        optimizer="AdamW",
        lr0=0.002,
        weight_decay=0.03,
        momentum=0.9,
        cos_lr=True,
        amp=True,

        # ─── Augmentation ────────────────────────────────
        # augment=True,
        # mosaic=0.6,
        # mixup=0.3,
        # hsv_h=0.015,
        # hsv_s=0.4,
        # hsv_v=0.4,
        # degrees=30.0,
        # translate=0.2,
        # scale=0.5,
        # shear=6.0,
        # perspective=0.1,
        # fliplr=0.5,
        # flipud=0.1,
        augment=True,
        mosaic=0.5,   # only 20% of the time
        mixup=0.3,    # only 10%
        hsv_h=0.01, 
        hsv_s=0.1, 
        hsv_v=0.1,
        degrees=25,   # smaller rotations
        translate=0.1,
        scale=0.3,
        shear=2,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        # ─── Training Control ─────────────────────────────
        patience=25,             # early stop if no val mAP improvement
        project="runs/train",
        name="VigilanceAI_V3",  # new experiment name
        exist_ok=True,
        plots=True,
        verbose=True,
        seed=42
    )

if __name__ == "__main__":
    main()
    