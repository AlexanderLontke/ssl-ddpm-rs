import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.data.util import transform_augment
from tqdm import tqdm

IMAGE_CONVERT = "RGB"
TIF_FILE_ENDING = ".tif"
UC_MERCED_CLASSES = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
]
STRING_TO_INT = {label: i for i, label in enumerate(UC_MERCED_CLASSES)}
INT_TO_STRING = {i: label for i, label in enumerate(UC_MERCED_CLASSES)}


class UCMerced(Dataset):
    def __init__(self, data_root, phase, split_ratios=None, **kwargs):
        super().__init__()
        assert phase in ["train", "test", "val"]
        if not split_ratios:
            split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
        # Load images from file system
        class_paths = glob.glob(data_root + "/*/")
        dataset_dict = {}
        indices = {k: {} for k in split_ratios.keys()}
        # Load all images into memory
        for path in tqdm(class_paths, desc="Loading Classes:"):
            class_label = path.split("/")[-2]
            class_images = []
            for image in glob.glob(path + f"/*{TIF_FILE_ENDING}"):
                class_images += [Image.open(image).convert(IMAGE_CONVERT)]
            dataset_dict[class_label] = class_images
            num_images = len(class_images)
            image_indices = list(range(num_images))
            for split, ratio in split_ratios.items():
                # Select indices for given split
                subset = np.random.choice(
                    image_indices, int(num_images * ratio), replace=False
                )
                # Store split indices
                indices[split][class_label] = subset
                # Remove indices for future splits
                image_indices = list(set(image_indices) - set(subset))
        # Build Index from split indices
        index = []
        indices = indices[phase]
        for class_name, index_values in indices.items():
            index += [(class_name, i) for i in index_values]
        # Declare as class variables
        self.index = index
        self.dataset_dict = dataset_dict
        self.phase = phase
        self.split_ratios = split_ratios

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        label, num = self.index[item]
        image = self.dataset_dict[label][num]
        # Use augment and norm from change detection paper
        image = transform_augment(image, split=self.phase, min_max=(-1, 1))
        return {"image": image, "L": STRING_TO_INT[label]}
