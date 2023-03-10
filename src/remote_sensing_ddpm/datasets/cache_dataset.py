import json

import torch
from pytorch_lightning import seed_everything

from remote_sensing_ddpm.evaluation.baselines.ddpm_cd import create_model
from remote_sensing_ddpm.datasets.uc_merced_land_use.uc_merced_dataset import UCMerced
from remote_sensing_ddpm.datasets.caching_dataset import CachingDataset


class NoneDict(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


if __name__ == '__main__':
    seed_everything(
        seed=42
    )

    training_parent_dataset = UCMerced(
        data_root="/netscratch2/alontke/master_thesis/data/UCMerced_LandUse/Images",
        phase="train",

    )

    validation_parent_dataset = UCMerced(
        data_root="/netscratch2/alontke/master_thesis/data/UCMerced_LandUse/Images",
        phase="val",

    )

    test_parent_dataset = UCMerced(
        data_root="/netscratch2/alontke/master_thesis/data/UCMerced_LandUse/Images",
        phase="test",
    )

    all_datasets = {
        "train": training_parent_dataset,
        "val": validation_parent_dataset,
        "test": test_parent_dataset
    }

    with open("config/baselines/ddpm-cd/classification.json") as config_file:
        opt = NoneDict(json.load(config_file))
    model = create_model(
        opt=opt,
    )
    for k, dataset in all_datasets.items():
        caching_dataset = CachingDataset(
            parent_dataset=dataset,
            model=model,
            feature_timesteps=opt["classification_model"]["time_steps"],
            image_key="image",
            label_key="L",
        )
        torch.save(
            obj=caching_dataset,
            f=f"/netscratch2/alontke/master_thesis/data/UCMerced_LandUse/CachedFeatures/{k}"
        )
