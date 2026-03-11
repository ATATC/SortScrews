import cv2

from sort_screws import SortScrewsDataset


DATASET_DIR: str = "SortScrews"

if __name__ == "__main__":
    dataset = SortScrewsDataset(DATASET_DIR, True)

