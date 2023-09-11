import os
import glob
from typing import Optional, List, Literal

from tqdm import tqdm

from torch import nn
import torch.utils.data as data

from remote_sensing_ddpm.datasets.dfc_2020.util import load_sample
from remote_sensing_ddpm.datasets.dfc_2020.constants import (
    DFC2020_CLASSES,
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
        no_savanna=False,
        use_s1=False,
        label_mode: Literal["classification", "segmentation", "regression"] = "segmentation",
        s1_augmentations: Optional[nn.Module] = None,
        s2_augmentations: Optional[nn.Module] = None,
        batch_augmentation: Optional[nn.Module] = None,
    ):
        """Initialize the dataset"""

        # inizialize
        super(DFC2020, self).__init__()

        # make sure parameters are okay
        self.use_s1 = use_s1
        self.s2_bands = s2_bands
        self.label_mode = label_mode
        assert subset in ["val", "test"]
        self.no_savanna = no_savanna

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

        s2_locations = glob.glob(os.path.join(path, "*.tif"), recursive=True)

        self.samples = []
        for s2_loc in tqdm(s2_locations, desc="[Load]"):
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            lc_loc = s2_loc.replace("_dfc_", "_lc_").replace("s2_", "dfc_")
            self.samples.append(
                load_sample(
                    {
                        "lc": lc_loc,
                        "s1": s1_loc,
                        "s2": s2_loc,
                        "id": os.path.basename(s2_loc),
                    },
                    self.use_s1,
                    self.s2_bands,
                    no_savanna=self.no_savanna,
                    igbp=False,
                    s1_augmentations=self.s1_augmentations,
                    s2_augmentations=self.s2_augmentations,
                    label_mode=self.label_mode,
                )
            )

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        print("loaded", len(self.samples), "samples from the dfc2020 subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        if self.batch_augmentation is not None:
            sample = self.batch_augmentation(sample)
        return sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)
