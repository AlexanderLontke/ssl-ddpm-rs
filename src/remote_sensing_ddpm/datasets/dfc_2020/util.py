from typing import Optional

import rasterio
import numpy as np

from torch import nn

from remote_sensing_ddpm.datasets.dfc_2020.constants import (
    S2_BANDS_HR,
    S2_BANDS_LR,
    S2_BANDS_MR,
    DFC2020_CLASSES,
)


# util function for reading s2 data
def load_s2(path, s2_bands, s2_augmentations: Optional[nn.Module] = None):
    bands_selected = s2_bands
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    if s2_augmentations is not None:
        s2 = s2_augmentations(s2)
    return s2


# util function for reading s1 data
def load_s1(path, s1_augmentations: Optional[nn.Module] = None):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    if s1_augmentations is not None:
        s1 = s1_augmentations(s1)
    return s1


# util function for reading lc data
def load_lc(path, no_savanna=False, igbp=True):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # convert IGBP to dfc2020 classes
    if igbp:
        lc = np.take(DFC2020_CLASSES, lc)
    else:
        lc = lc.astype(np.int64)

    # adjust class scheme to ignore class savanna
    if no_savanna:
        lc[lc == 3] = 0
        lc[lc > 3] -= 1

    # convert to zero-based labels and set ignore mask
    lc -= 1
    lc[lc == -1] = 255
    return lc


# util function for reading data from single sample
def load_sample(
    sample,
    use_s1,
    s2_bands,
    no_savanna=False,
    igbp=True,
    unlabeled=False,
    s1_augmentations=None,
    s2_augmentations=None,
):

    use_s2 = (len(s2_bands) > 0)

    # load s2 data
    if use_s2:
        img = load_s2(
            sample["s2"],
            s2_bands,
            s2_augmentations=s2_augmentations,
        )
    else:
        img = None

    # load s1 data
    if use_s1:
        s1_sample = load_s1(sample["s1"], s1_augmentations=s1_augmentations)
        if use_s2:
            img = np.concatenate((img, s1_sample), axis=0)
        else:
            img = s1_sample

    if img is None:
        raise ValueError("Neither Sentinel-1 nor Sentinel-2 data is used")

    # load label
    if unlabeled:
        return {"image": img, "id": sample["id"]}
    else:
        lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)
        return {"image": img, "label": lc, "id": sample["id"]}
