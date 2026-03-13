from os import PathLike
from typing import override

import cv2
import numpy as np
import torch
from torchvision.transforms import Resize
from mipcandy import HasDevice, Device, convert_logits_to_ids, auto_device

from sort_screws import Camera, ResNetPredictor


class Predictor(Camera, HasDevice):
    def __init__(self, experiment_folder: str | PathLike[str], *, device: Device = "cpu") -> None:
        Camera.__init__(self, 512)
        HasDevice.__init__(self, device)
        self.predictor: ResNetPredictor = ResNetPredictor(str(experiment_folder), (3, 512, 512),
                                                                      device=device)
        self.paused: bool = False
        self.resize: Resize = Resize(224)

    @override
    def job(self, frame: np.ndarray, roi: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        if not self.paused:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = torch.as_tensor(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), dtype=torch.float,
                                     device=self._device).permute(2, 0, 1)
            image = self.resize(image)
            logits = self.predictor.predict_image(image)
            class_id = convert_logits_to_ids(logits, channel_dim=0).item()
            cv2.putText(frame, f"Class: {class_id}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow("Camera Preview", frame)
        key = self.wait_key()
        if key == ord(" "):
            self.paused = not self.paused
        if key == ord("q"):
            return True
        return False


if __name__ == "__main__":
    device = auto_device()
    print(device)
    app = Predictor(f"trainer/{ResNetPredictor.__name__}/final", device=device)
    app.run()
