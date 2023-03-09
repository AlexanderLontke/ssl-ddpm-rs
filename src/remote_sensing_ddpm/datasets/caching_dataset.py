from typing import List

from torch.utils.data import Dataset

# Diffusion Model
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.models.model import DDPM


class CachingDataset(Dataset):
    def __init__(
        self,
        parent_dataset: Dataset,
        model: DDPM,
        feature_timesteps: List[int],
        image_key: str,
        label_key: str
    ):
        self.feature_timesteps = feature_timesteps
        self.items = []
        # Create cached representations
        for i in range(len(parent_dataset)):
            current_item = parent_dataset.__getitem__(i)
            assert all(hasattr(current_item, k) for k in [image_key, label_key])
            image = current_item[image_key]
            label = current_item[label_key]
            model.feed_data(current_item)
            feats = []
            for t in self.feature_timesteps:
                _, decoder_feats = model.get_single_representation(t=t)
                feats.append(decoder_feats)
            self.items += [{"feats": feats, image_key: image, label_key: label}]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]
