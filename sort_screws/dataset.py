from os.path import exists
from os import PathLike
from typing import override, Self
from json import load

import torch
from mipcandy import SupervisedDataset, JointTransform, Device
from pandas import read_csv


class SortScrewsDataset(SupervisedDataset[list[str] | list[int]]):
    def __init__(self, folder: str | PathLike[str], include_background: bool, *,
                 transform: JointTransform | None = None, device: Device = "cpu") -> None:
        csv_path = f"{folder}/labels.csv"
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = read_csv(csv_path, header="infer")
        if df.shape[1] < 2:
            raise ValueError(f"Expected [filename, class id] in {csv_path}, got shape {df.shape}")
        images, labels = [], []
        for _, row in df.iterrows():
            images.append(row.iloc[0])
            labels.append(int(row.iloc[1]))
        with open(f"{folder}/types.json") as f:
            types = load(f)
        self._num_classes: int = len(types)
        if include_background:
            images += ["case_120.png"] * 19
            images += ["case_241.png"] * 19
            labels = labels[:121] + [0] * 19 + labels[121:] + [0] * 19
        else:
            images.remove("case_120.png")
            images.remove("case_241.png")
            labels = labels[:120] + labels[121:-1]
        super().__init__(images, labels, transform=transform, device=device)
        self._folder: str = str(folder)
        self._include_background: bool = include_background

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        return self.do_load(f"{self._folder}/images/{self._images[idx]}", device=self._device)

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        return torch.tensor(self._labels[idx], dtype=torch.long, device=self._device)

    @override
    def construct_new(self, images: list[str], labels: list[int]) -> Self:
        new = SortScrewsDataset(self._folder, self._include_background, transform=self._transform, device=self._device)
        new._images = images
        new._labels = labels
        return new
