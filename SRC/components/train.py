
from ultralytics import YOLO

def main():
    #baseline model
    model = YOLO("models/yolo11s.pt")
    
    
    #training model
    model.train(
        data="data/data.yaml",    # path to your data.yaml (train/val splits & class names)
        epochs=50,                # total epochs
        batch=8,                 # images per batch
        imgsz=512,                # resize to 640×640
        augment=True,             # enable built‑in Mosaic, MixUp, HSV, etc.
        patience=5,               # early stop after 5 epochs w/o val mAP improvement
        project="runs/train",     # top‑level folder for outputs
        name="VigilanceAI_V1",            # subfolder (→ runs/train/phase1)
        exist_ok=True,            # overwrite if that folder already exists
        plots=True,               # save training/validation metric plots
        device="cuda",              # to train on gpu
        verbose=True,               # for notes
        seed=42,                   #just random see
        optimizer="AdamW",
        lr0=0.001,                      # Start higher than SGD
        weight_decay=0.03,      # Critical for AdamW
        momentum=0.9,               # Works with AdamW in YOLO
        cos_lr=True
)


#using name == main to avoid runtime error fork vs spawn which windows uses
if __name__ == "__main__":
    main()
    