from pathlib import Path
from typing import Dict

# FFCV
from ffcv.writer import DatasetWriter

# PyTorch
from torch.utils.data import Dataset


# Object instantiation
from lit_diffusion.util import instantiate_python_class_from_string_config


def write_dataset_to_ffcv_format(dataset_config: Dict, ffcv_writer_config: Dict, *args, **kwargs):
    dataset: Dataset = instantiate_python_class_from_string_config(
        class_config=dataset_config,
        verbose=True
    )

    ffcv_writer: DatasetWriter = instantiate_python_class_from_string_config(
        class_config=ffcv_writer_config,
        verbose=True
    )
    # Write dataset to .beton format
    ffcv_writer.from_indexed_dataset(dataset)


if __name__ == '__main__':
    import argparse
    import yaml
    # Add run arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to config.yaml",
        required=False,
    )

    # Parse run arguments
    args = parser.parse_args()

    # Load config file
    config_file_path = args.config
    with config_file_path.open("r") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)

    # Run main function
    write_dataset_to_ffcv_format(
        **config,
    )
