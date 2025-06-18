
from ultralytics import YOLO

def main():
    # Start from the official YOLOv11s pretrained weights
    model = YOLO("models/yolov9s.pt")

    model.train(
    data="config/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    device="cuda",

    optimizer="AdamW",
    lr0=0.0003,        # smaller base LR
    lrf=0.005,         # gentler decay
    weight_decay=0.0005,
    momentum=0.9,
    cos_lr=True,
    amp=True,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.01,
    cls=1.5,
    box=7.5,
    dfl=2,

    augment=True,      # light augment
    mosaic=0.3,
    copy_paste=0.3,
    copy_paste_mode="mixup",
    mixup=0.1,
    cutmix=0.1,
    hsv_h=0.1,
    hsv_s=0.15,
    hsv_v=0.15,
    degrees=15,
    scale=0.2,
    shear=0.5,
    fliplr=0.5,
    flipud=0.05,
    erasing=0.2,

    patience=10,       # earlier stop
    close_mosaic=10,
    project="runs/train",
    name="PPE_SAFETY_AI_final1234",
    exist_ok=True,
    plots=True,
    verbose=True,
    seed=42,
)

if __name__ == "__main__":
    main()

    """
    validating runs\train\PPE_SAFETY_AI_final\weights\best.pt...
Ultralytics 8.3.155  Python-3.10.16 torch-2.7.0+cu126 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
YOLOv9s summary (fused): 197 layers, 7,168,636 parameters, 0 gradients, 26.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:02<00:00,  1.80it/s]
                   all        120        330      0.903      0.804      0.878      0.527
                Gloves         21         46      0.903       0.63      0.694      0.355
               Hardhat         39         81      0.986      0.926      0.981      0.593
                Person         85        152       0.92      0.836       0.94      0.609
           Safety Vest         26         51        0.8      0.824      0.895      0.551
Speed: 0.4ms preprocess, 13.1ms inference, 0.0ms loss, 2.3ms postprocess per image
Results saved to runs\train\PPE_SAFETY_AI_final

"""