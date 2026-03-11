from os.path import exists

from torch.utils.data import DataLoader
from mipcandy import download_dataset

from sort_screws import SortScrewsDataset, EfficientNetTrainer

if __name__ == "__main__":
    if not exists("SortScrews") and input("Dataset not found, download? (y/n) >>>") == "y":
        download_dataset("atatc/ut/esc102/SortScrews", "SortScrews")
    dataset = SortScrewsDataset("SortScrews")
    train, val = dataset.fold(fold="all")
    train_dl = DataLoader(train, 16, True, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_dl = DataLoader(val, 1, False)
    trainer = EfficientNetTrainer("trainer", train_dl, val_dl, recoverable=False)
    trainer.num_classes = dataset.num_classes
    trainer.train(100, seed=42, compile_model=False)
