from typing import Optional, List

import yaml
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from lit_diffusion.util import instantiate_python_class_from_string_config
from remote_sensing_ddpm.run_downstream_tasks import fuse_backbone_and_downstream_head_config, get_best_checkpoints, create_wandb_run_name

from lit_diffusion.constants import (
    PL_MODULE_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    DEVICE_CONFIG_KEY,
    PYTHON_KWARGS_CONFIG_KEY,
)

PROJECT_KEY = "name"

def get_model_and_test_dataloader_from_configs(backbone_config_path: Path, downstream_head_config_path: Path, checkpoints_root_path: Path, test_beton_file: Optional[str] = None, custom_batch_size: Optional[int] = None):
    
    with backbone_config_path.open("r") as backbone_config_file:
        backbone_config = yaml.safe_load(backbone_config_file)
    
    with downstream_head_config_path.open("r") as downstream_head_config_file:
        downstream_head_config = yaml.safe_load(downstream_head_config_file)
        
    complete_config = fuse_backbone_and_downstream_head_config(
        backbone_config=backbone_config,
        downstream_head_config=downstream_head_config,
    )
    
    wandb_run_name = create_wandb_run_name(
        backbone_name=backbone_config_path.name,
        downstream_head_name=downstream_head_config_path.name,
    )
    wandb_sub_project_name = complete_config[PROJECT_KEY]
    
    best_checkpoints = get_best_checkpoints(
        wandb_run_name=wandb_run_name,
        complete_config=complete_config,
        wandb_sub_project_name=wandb_sub_project_name, 
        through_callback=True,
        checkpoints_root_path=checkpoints_root_path,
    )
    checkpoint_path = checkpoints_root_path / best_checkpoints[0]
    pl_module = instantiate_python_class_from_string_config(
        class_config=complete_config[PL_MODULE_CONFIG_KEY]
    )
    pl_module = pl_module.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=complete_config[DEVICE_CONFIG_KEY],
        downstream_model=pl_module.downstream_model,
        learning_rate=pl_module.learning_rate,
        loss=pl_module.loss,
        target_key=pl_module.target_key,
        validation_metrics=pl_module.validation_metrics,
    )
    pl_module = pl_module.eval()

    # Create test dataloader config
    test_dataloader_config = copy.deepcopy(
        complete_config[VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY]
    )
    if test_beton_file is not None:
        test_dataloader_config[PYTHON_KWARGS_CONFIG_KEY]["fname"] = test_beton_file
    if custom_batch_size is not None:
        test_dataloader_config[PYTHON_KWARGS_CONFIG_KEY]["batch_size"] = custom_batch_size
    test_dataloader = instantiate_python_class_from_string_config(
        class_config=test_dataloader_config
    )
    return pl_module, test_dataloader
    


def visualize_batched_segmentation_sample(image, y_hat, y, num_classes, value_to_class_mapping, figsize=(6, 12), dimming_factor=3.0):
    # Setup plot grid
    b, *_ = image.shape
    fig, axs = plt.subplots(b, 3)

    # Set column labels
    cols = ["Input image (RGB)", "Prediction", "Label"]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    # Reformat RGB image
    image = image[:, [0,1,2], :, :].permute(0, 2, 3, 1).clip(-1, 1) 
    image = ((image + 1) / 2) * dimming_factor
    image = image.clip(0,1).detach().cpu().numpy()

    y_hat = y_hat.argmax(dim=1)
    y_hat, y = (x.detach().cpu().numpy() for x in (y_hat, y))

    shared_cmap=mpl.colormaps[mpl.rcParams["image.cmap"]]
    shared_norm=mpl.colors.Normalize(vmin=0, vmax=int(num_classes-1))
    # Plot images
    for i in range(b):
        ax_row = axs[i]
        for ax, img in zip(ax_row, [image[i], y_hat[i], y[i]]):
            im = ax.imshow(img, cmap=shared_cmap, norm=shared_norm)
            ax.axis("off")

    # Create shared legend
    land_cover_classes = list(range(num_classes))
    patches = [mpatches.Patch(color=shared_cmap(shared_norm(lc_class)), label=value_to_class_mapping[lc_class]) for lc_class in land_cover_classes]  
    fig.legend(handles=patches, loc=2, bbox_to_anchor=(0.9, 0.9),)
