from typing import List
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

# Diffusion Model
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.models.model import DDPM


class CachingDataset(Dataset):
    def __init__(
        self,
        parent_dataset: Dataset,
        model: DDPM,
        feature_timesteps: List[int],
        image_key: str,
        label_key: str,
    ):
        self.feature_timesteps = feature_timesteps
        self.items = []

        data_loader = DataLoader(
            dataset=parent_dataset,
            batch_size=10,
            num_workers=4,
        )
        # Create cached representations
        for _, current_item in tqdm(enumerate(data_loader), desc="Computing Representations", total=len(data_loader)):
            # assert all(hasattr(current_item, k) for k in [image_key, label_key])
            image = current_item[image_key]
            label = current_item[label_key]
            model.feed_data(current_item)
            feats = []
            for t in self.feature_timesteps:
                _, decoder_feats = model.get_single_representation(t=t)
                feats.append([f.cpu() for f in decoder_feats])
                del decoder_feats
            
            self.items += [{"feats": feats, image_key: image, label_key: label}]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]
