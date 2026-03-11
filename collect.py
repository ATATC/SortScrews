from csv import writer
from json import dump
from os import makedirs
from typing import override

import cv2
import numpy as np

from sort_screws import Camera

DATASET_DIR: str = "SortScrews"


class Collector(Camera):
    def __init__(self) -> None:
        super().__init__(256)
        self.images_dir: str = f"{DATASET_DIR}/images"
        self.csv_path: str = f"{DATASET_DIR}/labels.csv"
        makedirs(self.images_dir, exist_ok=True)
        with open(self.csv_path, "w") as f:
            f.write("filename,class\n")
        # runtime
        self.class_id: int = 0
        self.num_cases: int = 0

    @override
    def job(self, frame: np.ndarray, roi: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {self.class_id}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Num cases: {self.num_cases}", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                    2,
                    cv2.LINE_AA)
        cv2.imshow("Camera Preview", frame)
        key = self.wait_key()
        if key == ord("c"):
            filename = f"case_{self.num_cases:03d}.png"
            cv2.imwrite(f"{self.images_dir}/{filename}", roi)
            with open(self.csv_path, "a", newline="") as f:
                w = writer(f)
                w.writerow([filename, self.class_id])
            print(f"Saved {filename} (class {self.class_id})")
            self.num_cases += 1
        elif key == ord("q"):
            return True
        elif ord("0") <= key <= ord("9"):
            self.class_id = key - ord("0")
        return False


if __name__ == "__main__":
    app = Collector()
    app.run()
    with open(f"{DATASET_DIR}/types.json", "w") as f:
        dump([
            {"class_id": 0, "description": "background / no screw"},
            {"class_id": 1, "description": "flat 1.5cm"},
            {"class_id": 2, "description": "round 2.5cm"},
            {"class_id": 3, "description": "flat 3.0cm"},
            {"class_id": 4, "description": "flat 3.5cm"},
            {"class_id": 5, "description": "flat 6.0cm"},
            {"class_id": 6, "description": "flat 7.5cm"}
        ], f)
