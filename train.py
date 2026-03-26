from os import rename
from os.path import exists

from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from mipcandy import download_dataset, auto_device, MONAITransform, JointTransform

from sort_screws import SortScrewsDataset, ConvNeXtTrainer

if __name__ == "__main__":
    if not exists("SortScrews") and input("Dataset not found, download? (y/n) >>>") == "y":
        download_dataset("atatc/ut/esc102/SortScrews", "SortScrews")
    transforms = JointTransform(image_only=MONAITransform(Resize(224)))
    train = SortScrewsDataset("SortScrews", True, transform=transforms)
    val = SortScrewsDataset("SortScrews/validation", True, transform=transforms)
    if train.num_classes != val.num_classes:
        raise ValueError(f"Number of classes mismatch: train={train.num_classes}, val={val.num_classes}")
    train_dl = DataLoader(train, 16, True, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_dl = DataLoader(val, 1, False)
    trainer = ConvNeXtTrainer("trainer", train_dl, val_dl, recoverable=False, device=auto_device())
    trainer.num_classes = train.num_classes
    trainer.train(100, seed=42, compile_model=False, early_stop_tolerance=100)
    target = f"trainer/{trainer.__class__.__name__}/final"
    experiment_folder = trainer.experiment_folder()
    if exists(target):
        print(f"{trainer} already exists, {experiment_folder} kept unchanged")
    else:
        rename(experiment_folder, target)
