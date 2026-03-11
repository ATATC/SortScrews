from os.path import exists
from os import PathLike
from typing import override, Self
from json import load

import torch
from mipcandy import SupervisedDataset, JointTransform, Device
from pandas import read_csv


class SortScrewsDataset(SupervisedDataset[list[str] | list[int]]):
    def __init__(self, folder: str | PathLike[str], transform: JointTransform | None = None, device: Device = "cpu") -> None:
        csv_path = f"{folder}/labels.csv"
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = read_csv(csv_path, header="infer")
        if df.shape[1] < 2:
            raise ValueError(f"Expected [filename, class id] in {csv_path}, got shape {df.shape}")
        images, labels = [], []
        for _, row in df.iterrows():
            image_path = f"{folder}/images/{row.iloc[0]}"
            class_id = int(row.iloc[1])
            if not exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            images.append(image_path)
            labels.append(class_id)
        with open(f"{folder}/types.json") as f:
            types = load(f)
        self._num_classes: int = len(types)
        super().__init__(images, labels, transform=transform, device=device)
        self._folder: str = str(folder)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        return self.do_load(self._images[idx], device=self._device)

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        return torch.tensor(self._labels[idx], dtype=torch.long, device=self._device)

    @override
    def construct_new(self, images: list[str], labels: list[int]) -> Self:
        new = SortScrewsDataset(self._folder, transform=self._transform, device=self._device)
        new._images = images
        new._labels = labels
        return new
