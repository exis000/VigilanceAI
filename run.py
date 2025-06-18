from src.inference import infer
from ultralytics import YOLO
import cv2
import os

def update(file_path: str, root: str):
    with open(file_path, "r") as file:
        paths = file.readlines()

    updated_paths = list[str]()
    for path in paths:
        path = path.strip()
        updated_paths.append(root + path + "\n")

    new_file_name = "updated_" + os.path.basename(file_path)
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(new_file_path, "w") as file:
        file.writelines(updated_paths)

    print("Updated file created successfully: " + new_file_path)


def main2():
    root = r"C:/Users/exis/python files exis/projects/proffesional projects/vigilanceAI"
    update(file_path=r"data/SH17_Dataset_for_PPE_Detection/train_files.txt", root=root)
    update(file_path=r"data/SH17_Dataset_for_PPE_Detection/val_files.txt", root=root)



def main():
    # 1) Load your trained model
    model = YOLO("models/person_v1.pt")

    # 2) Read the input image
    img_path = "demos/examples/worker1.png"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")

    # 3) Run prediction (class 0 = person, conf threshold 0.15)
    #    We pass the image directly instead of a path so we get the results object back.
    results = model.predict(
        source=img,
        classes=[0],
        conf=0.15,
        iou=0.55,
        max_det=50,
        verbose=False,  # suppress console logs except the count
    )[0]

    # 4) Extract boxes, confidences, and class indices
    boxes = results.boxes.xyxy.cpu().numpy()    # shape (N,4)
    confs = results.boxes.conf.cpu().numpy()    # shape (N,)
    clses = results.boxes.cls.cpu().numpy()     # shape (N,)

    # 5) Print how many persons were detected
    num_persons = len(boxes)
    print(f"Detected {num_persons} person(s) with conf >= 0.15")

    # 6) Draw boxes & labels on the image
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clses):
        # Convert to ints
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        # Draw a green rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put a text label with confidence
        label = f"person {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 7) Display the result in a window (skip if headless)
    try:
        cv2.imshow("Person Detection", img)
        print("Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("GUI not available, skipping cv2.imshow()")

    # 8) Save the annotated image
    out_path = "demos/results/worker1_out.png"
    cv2.imwrite(out_path, img)
    print(f"Annotated image saved to {out_path}")

    
if __name__ == "__main__":
    #infer.run_inference(src_video="demos/examples/Construction_worker.mp4", out_video="demos/results/result3.mp4")
    #main()
    main2()