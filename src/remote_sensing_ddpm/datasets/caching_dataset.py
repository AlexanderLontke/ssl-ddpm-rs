from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# Diffusion Model
from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.model import DDPM


class CachingDataset:
    def __init__(
        self,
        parent_dataset: Dataset,
        model: DDPM,
        feature_timesteps: List[int],
        image_key: str,
        label_key: str,
        data_root: str,
        label_to_nl_dict: Dict,
    ):
        self.feature_timesteps = feature_timesteps

        data_loader = DataLoader(
            dataset=parent_dataset,
            batch_size=10,
            num_workers=4,
        )
        # Create cached representations
        for _, current_item in tqdm(
            enumerate(data_loader),
            desc="Computing Representations",
            total=len(data_loader),
        ):
            images = current_item[image_key]
            labels = current_item[label_key]
            batch_size, *_ = labels.shape
            model.feed_data(current_item)
            feats = []
            for t in self.feature_timesteps:
                _, decoder_feats = model.get_single_representation(t=t)
                feats.append([f.cpu() for f in decoder_feats])
                del decoder_feats
            counts = {}
            for i in range(batch_size):
                current_image = images[i]
                current_label = labels[i].item()
                current_feats = []
                for j in range(len(self.feature_timesteps)):
                    current_feats.append(feats[j][i])
                current_root_path = (
                    Path(data_root) / f"{label_to_nl_dict[current_label]}"
                )
                current_root_path.mkdir(parents=True, exist_ok=True)
                file_number = 0
                if current_label in counts.keys():
                    file_number = counts[current_label]
                    counts[current_label] = file_number + 1
                else:
                    counts[current_label] = 1

                torch.save(
                    obj=current_image,
                    f=current_root_path / f"image_{file_number}.pth",
                )
                torch.save(
                    obj=current_feats,
                    f=current_root_path / f"feats_{file_number}.pth",
                )
