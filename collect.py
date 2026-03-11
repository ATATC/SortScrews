from csv import writer
from dataclasses import dataclass
from os import makedirs

import cv2


@dataclass
class Runtime(object):
    class_id: int = 0
    num_cases: int = 0


CAMERA_DEVICE_ID: int = 0
DATASET_DIR: str = "SortScrews"
BOX_SIZE: int = 256
EXPORT_SIZE: int = 256

if __name__ == "__main__":
    runtime = Runtime()
    images_dir = f"{DATASET_DIR}/images"
    csv_path = f"{DATASET_DIR}/labels.csv"
    makedirs(images_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("filename,class\n")
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {CAMERA_DEVICE_ID}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        cx = w // 2
        cy = h // 2
        half = BOX_SIZE // 2
        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half
        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"Class: {runtime.class_id}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(preview, f"Num cases: {runtime.num_cases}", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("Camera Preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (EXPORT_SIZE, EXPORT_SIZE), interpolation=cv2.INTER_AREA)
            filename = f"case_{runtime.num_cases:03d}.png"
            cv2.imwrite(f"{images_dir}/{filename}", crop)
            with open(csv_path, "a", newline="") as f:
                w = writer(f)
                w.writerow([filename, runtime.class_id])
            print(f"Saved {filename} (class {runtime.class_id})")
            runtime.num_cases += 1
        elif key == ord("q"):
            break
        elif ord("0") <= key <= ord("9"):
            runtime.class_id = key - ord("0")
    cap.release()
    cv2.destroyAllWindows()
