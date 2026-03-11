from csv import writer
from os import PathLike
from typing import override

import cv2
import numpy as np

from sort_screws.camera import Camera
from sort_screws.dataset import SortScrewsDataset


class Editor(Camera):
    def __init__(self, dataset_dir: str | PathLike[str], roi_size: int, *, resize: int | None = None,
                 device_id: int = 0) -> None:
        super().__init__(roi_size, resize=resize, device_id=device_id)
        self.dataset_dir: str = str(dataset_dir)
        self.dataset: SortScrewsDataset = SortScrewsDataset(dataset_dir, True)
        self.idx: int = 0
        self.refresh: bool = True
        self.editing: bool = False

    @override
    def job(self, frame: np.ndarray, roi: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        raise NotImplementedError

    @override
    def run(self) -> None:
        while True:
            if self.refresh:
                image = self.dataset.image(self.idx)
                label = getattr(self.dataset, "_labels")[self.idx]
                frame = image.numpy()
                cv2.putText(frame, f"Class: {label}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f"Editing: {self.editing}", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                            2,
                            cv2.LINE_AA)
                cv2.imshow("Dataset Preview", frame)
            key = self.wait_key()
            self.refresh = True
            if key == ord("e"):
                self.editing = not self.editing
            elif key == ord("["):
                self.idx = (self.idx - 1) % len(self.dataset)
            elif key == ord("]"):
                self.idx = (self.idx + 1) % len(self.dataset)
            elif ord("0") <= key <= ord("9") and self.editing:
                getattr(self.dataset, "_labels")[self.idx] = key - ord("0")
                with open(f"{self.dataset_dir}/labels.csv", "w") as f:
                    w = writer(f)
                    w.writerows([(
                        getattr(self.dataset, "_images")[idx], getattr(self.dataset, "_labels")[idx]
                    ) for idx in range(len(self.dataset))])
                self.editing = False
            elif key == ord("q"):
                break
            else:
                self.refresh = False
        cv2.destroyAllWindows()
