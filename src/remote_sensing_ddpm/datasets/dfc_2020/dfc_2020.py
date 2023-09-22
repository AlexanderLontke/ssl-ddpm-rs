import os
import glob
from typing import Optional, List, Literal

import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.utils.data as data

from remote_sensing_ddpm.datasets.dfc_2020.util import load_sample, get_class_fractions_from_segmentation_map
from remote_sensing_ddpm.datasets.dfc_2020.constants import (
    DFC2020_CLASSES,
    LABEL_KEY,
)


# calculate number of input channels
def get_ninputs(use_s1, s2_bands):
    n_inputs = len(s2_bands)
    if use_s1:
        n_inputs += 2
    return n_inputs


class DFC2020(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(
        self,
        path,
        s2_bands: List[int],
        subset="val",
        no_savanna: bool =False,
        no_snow_and_savanna: bool = False,
        use_s1: bool =False,
        load_on_the_fly: bool = True,
        label_dtype: str = "float",
        label_mode: Literal["classification", "segmentation", "regression"] = "segmentation",
        s1_augmentations: Optional[nn.Module] = None,
        s2_augmentations: Optional[nn.Module] = None,
        batch_augmentation: Optional[nn.Module] = None,
        samples_subset_path: Optional[str] = None,
    ):
        """Initialize the dataset"""

        # inizialize
        super(DFC2020, self).__init__()

        # make sure parameters are okay
        self.load_on_the_fly = load_on_the_fly
        self.use_s1 = use_s1
        self.s2_bands = s2_bands
        self.label_dtype = label_dtype
        self.label_mode = label_mode
        assert subset in ["val", "test"]
        self.no_savanna = no_savanna
        self.no_snow_and_savanna = no_snow_and_savanna
        self.num_classes = 10
        if self.no_savanna:
            self.num_classes -= 1
        if self.no_snow_and_savanna:
            self.num_classes -= 2

        # store augmentations if provided
        self.s1_augmentations = s1_augmentations
        self.s2_augmentations = s2_augmentations
        self.batch_augmentation = batch_augmentation

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, s2_bands)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(DFC2020_CLASSES)

        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        if subset == "val":
            path = os.path.join(path, "ROIs0000_validation", "s2_0")
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")

        if samples_subset_path is not None:
            samples_subset = pd.read_csv(samples_subset_path)
            s2_locations = [os.path.join(path, x) for x in samples_subset["id"].tolist()]
        else:
            s2_locations = glob.glob(os.path.join(path, "*.tif"), recursive=True)

        self.samples = []
        for s2_loc in tqdm(s2_locations, desc="[Load]"):
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            lc_loc = s2_loc.replace("_dfc_", "_lc_").replace("s2_", "dfc_")
            sample = {
                "lc": lc_loc,
                "s1": s1_loc,
                "s2": s2_loc,
                "id": os.path.basename(s2_loc),
            }
            if not self.load_on_the_fly:
                sample = load_sample(
                    sample,
                    self.use_s1,
                    self.s2_bands,
                    no_savanna=self.no_savanna,
                    igbp=False,
                    no_snow_and_savanna=self.no_snow_and_savanna,
                    s1_augmentations=self.s1_augmentations,
                    s2_augmentations=self.s2_augmentations,
                )
            self.samples.append(
                sample
            )

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        print("loaded", len(self.samples), "samples from the dfc2020 subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        if self.load_on_the_fly:
            sample = load_sample(
                    sample,
                    self.use_s1,
                    self.s2_bands,
                    no_savanna=self.no_savanna,
                    no_snow_and_savanna=self.no_snow_and_savanna,
                    igbp=False,
                    s1_augmentations=self.s1_augmentations,
                    s2_augmentations=self.s2_augmentations,
                )
        if self.batch_augmentation is not None:
            sample = self.batch_augmentation(sample)

        # get correct label
        lc = sample[LABEL_KEY]
        if self.label_mode == "classification":
            lc = get_class_fractions_from_segmentation_map(lc, num_classes=self.num_classes) > 0.05
        elif self.label_mode == "regression":
            lc = get_class_fractions_from_segmentation_map(lc, num_classes=self.num_classes)
        elif self.label_mode == "segmentation":
            lc = lc
        else:
            raise NotImplementedError(f"label mode {self.label_mode} not implemented")
        lc = torch.Tensor(lc)
        sample[LABEL_KEY] = lc.type(getattr(torch, self.label_dtype))
        return sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)
