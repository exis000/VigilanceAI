from ultralytics import YOLO

def evaluate_checkpoint(ckpt_path, data_yaml):
    model = YOLO(ckpt_path)
    # Run validation
    results = model.val(data=data_yaml, batch=8, device="cuda", plots=False)

    # Overall metrics
    m = results.results_dict
    print(f"\nOverall →  Precision: {m['metrics/precision(B)']:.3f}"
          f", Recall: {m['metrics/recall(B)']:.3f}"
          f", mAP50: {m['metrics/mAP50(B)']:.3f}"
          f", mAP@[.50:.95]: {m['metrics/mAP50-95(B)']:.3f}\n")

    # Per‑class AP@50 and AP@[.50:.95]
    metric = results.box  # Metric object
    ap50_per_cls = metric.ap50()      # list of AP@.50 for each class
    ap_all_per_cls = metric.maps()    # list of AP@[.50:.95] for each class

    for idx, name in results.names.items():
        print(f"Class {idx:2d} ({name:12s}) →"
              f" AP@50 = {ap50_per_cls[idx]:.3f},"
              f" AP@[.50:.95] = {ap_all_per_cls[idx]:.3f}")

if __name__ == "__main__":
    ckpt = "models/VigilanceAI_V2.pt"
    yaml = "data/data.yaml"
    evaluate_checkpoint(ckpt, yaml)