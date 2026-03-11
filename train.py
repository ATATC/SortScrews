from os import rename
from os.path import exists

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomAffine, ColorJitter, Normalize
from mipcandy import download_dataset, auto_device, JointTransform

from sort_screws import SortScrewsDataset, EfficientNetTrainer

if __name__ == "__main__":
    if not exists("SortScrews") and input("Dataset not found, download? (y/n) >>>") == "y":
        download_dataset("atatc/ut/esc102/SortScrews", "SortScrews")
    dataset = SortScrewsDataset("SortScrews", True)
    train, val = dataset.fold(fold="all")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = Compose([
        RandomAffine(degrees=12, translate=(.08, .08), scale=(.9, 1.1)),
        ColorJitter(brightness=.2, contrast=.2, saturation=.1, hue=.02),
        Normalize(mean=mean, std=std)
    ])
    val_tf = Compose([Normalize(mean=mean, std=std)])
    train.set_transform(JointTransform(image_only=train_tf))
    val.set_transform(JointTransform(image_only=val_tf))
    train_dl = DataLoader(train, 16, True, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_dl = DataLoader(val, 1, False)
    trainer = EfficientNetTrainer("trainer", train_dl, val_dl, recoverable=False, device=auto_device())
    trainer.num_classes = dataset.num_classes
    trainer.train(100, seed=42, compile_model=False)
    target = "trainer/EfficientNetTrainer/final"
    experiment_folder = trainer.experiment_folder()
    if exists(target):
        print(f"{trainer} already exists, {experiment_folder} kept unchanged")
    else:
        rename(experiment_folder, target)
