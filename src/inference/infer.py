# src/inference/infer.py
import cv2
from ultralytics import YOLO
from src.utils import xyxy_iou

# Load trained model
MODEL_PATH = "runs/train/VigilanceAI_V3/weights/best.pt"

PERSON_CONF = 0.05   # lower to catch all people
PPE_CONF    = 0.15   # for PPE classes
PPE_IOU_TH  = 0.30   # overlap threshold to “attach” PPE to a person
PADDING     = 20     # px padding around person bbox for cropping

def run_inference(src_video: str, out_video: str):
    """
    Two-stage PPE compliance on a video file.

    1) Detect people in each frame.
    2) For each person, crop and detect PPE.
    3) Draw boxes and flag NON-COMPLIANT people.

    Args:
        src_video: Path to input video (mp4, avi, etc.).
        out_video: If set, path to save annotated output video.
    """
    # Load your YOLO model once
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(src_video)
    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_video, fourcc, 20, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ─── Stage 1: Person Detection ─────────────────────────
        people_boxes = model.predict(
            frame,
            classes=[0],            # only “person”
            conf=PERSON_CONF,
            iou=0.45,
            max_det=50
        )[0].boxes.xyxy.cpu().numpy()  # Nx4 array of [x1,y1,x2,y2]

        # ─── Stage 2: PPE Detection in each person crop ────────
        ppe_detections = []  # will store (box, cls, conf)
        for (x1, y1, x2, y2) in people_boxes:
            # add padding and clamp
            x1i = max(int(x1) - PADDING, 0)
            y1i = max(int(y1) - PADDING, 0)
            x2i = min(int(x2) + PADDING, frame.shape[1])
            y2i = min(int(y2) + PADDING, frame.shape[0])

            crop = frame[y1i:y2i, x1i:x2i]
            res  = model.predict(
                crop,
                classes=[1,2,3,4],  # {1:hardhat,2:glasses,3:vest,4:gloves}
                conf=PPE_CONF,
                iou=0.45,
                max_det=20
            )[0]

            # translate crop‐local boxes back to full frame
            for box, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy(),
                                      res.boxes.cls.cpu().numpy()):
                bx1, by1, bx2, by2 = box
                abs_box = (bx1 + x1i, by1 + y1i, bx2 + x1i, by2 + y1i)
                ppe_detections.append((abs_box, int(cls), float(conf)))

        # ─── DRAW & CHECK COMPLIANCE ───────────────────────────
        for (x1, y1, x2, y2) in people_boxes:
            # draw person
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

            # gather which PPE classes overlap this person
            has_ppe = {1:False, 2:False, 3:False, 4:False}
            for (bx1,by1,bx2,by2), cls, conf in ppe_detections:
                if xyxy_iou((x1,y1,x2,y2), (bx1,by1,bx2,by2)) > PPE_IOU_TH:
                    has_ppe[cls] = True
                    # draw PPE box
                    cv2.rectangle(frame, (int(bx1),int(by1)), (int(bx2),int(by2)),
                                  (255,0,0), 2)
                    cv2.putText(frame,
                                f"{model.names[cls]} {conf:.2f}",
                                (int(bx1),int(by1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,0,0), 1)

            # if any PPE missing, flag
            if not all(has_ppe.values()):
                cv2.putText(frame, "NON-COMPLIANT",
                            (int(x1),int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)

        # write out
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()