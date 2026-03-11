# SortScrew

SortScrew is a dataset for screw classification. We collected 242 images on 6 types of screws.

## Citation

```bibtex
```

## Download

You can download the dataset from
[Project Neura's Central Data Server (CDS)](https://cds.projectneura.org/atatc/ut/esc102/SortScrews.zip).

Alternatively, you can download the dataset using MIP Candy:

```python
from mipcandy import download_dataset

download_dataset("atatc/ut/esc102/SortScrews", "dir/to/save/dataset")
```

## Class Indices

We rank the screws by their lengths. Class indices are assigned from 1 to 6, from left to right.

Class 0 is reserved for the background.

![types of screws](assets/types.png)

## Customization

You could use "collect.py" to collect your own dataset. Press "0" to "9" to set the current class id, and then press "c"
to capture the image. Press "q" to quit.