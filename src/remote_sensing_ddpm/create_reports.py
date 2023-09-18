from typing import Dict, List, Optional, Callable, Literal
from functools import partial
from pathlib import Path

# Matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Remote Sensing DDPM
from remote_sensing_ddpm.run_downstream_tasks import (
    create_wandb_run_name,
    LABEL_FRACTION_PATHS,
)
from remote_sensing_ddpm.utils.viz_style import set_matplotlib_style

# WandB to paper imports
from from_wandb_to_paper.tables.metrics_table import get_metrics_table
from from_wandb_to_paper.latex.tables import metrics_table_to_latex
from from_wandb_to_paper.figures.metric_errorbar import (
    visualize_single_metrics_table_metric,
)
from from_wandb_to_paper.figures.label_fractions import get_label_fraction_figure
from from_wandb_to_paper.util.data_aggregation import calculate_class_weighted_mean

CLASS_WEIGHTED_AVERAGE_NAME = "Class-Weighted Average"
UNIFORM_AVERAGE_NAME = "Uniform Average"

TEST_SET_SEGMENTATION_CLASS_FRACTIONS = {
    "Tree cover": 0.4300583543210937,
    "Shrubland": 0.01087377054654035,
    "Grassland": 0.2075682758077903,
    "Cropland": 0.13916261817540043,
    "Built-up": 0.016301759871402245,
    "Bare": 0.0011900179349951731,
    "Snow and Ice": 0.0,
    "Permanenet water bodies": 0.1891719553437699,
    "Herbaceous wetland": 0.005394617593467072,
    "Mangroves": 0.0,
    "Moss and lichen": 0.0002786304055407682,
}

TEST_SET_CLASSIFICATION_CLASS_FRACTIONS = {
    "Tree cover": 0.34617318327725566,
    "Shrubland": 0.021386858490930888,
    "Grassland": 0.28912866921916697,
    "Cropland": 0.16718876899872376,
    "Built-up": 0.033917314460300885,
    "Bare": 0.002088409328228333,
    "Snow and Ice": 0.0,
    "Permanenet water bodies": 0.1288239161542329,
    "Herbaceous wetland": 0.010790114862513052,
    "Mangroves": 0.0,
    "Moss and lichen": 0.0005027652086475615,
}

DFC_2020_TEST_SET_SEGMENTATION_CLASS_FRACTIONS = {
    "Forest": 0.25629095163806553,
    "Shrubland": 0.05897753618478072,
    "Grassland": 0.10070588761050442,
    "Wetlands": 0.021144099930663893,
    "Croplands": 0.19467160144739123,
    "Urban_Built-up": 0.10820969297105218,
    "Barren": 0.026166282067949385,
    "Water": 0.23383394814959266,
}

DFC_2020_TEST_SET_CLASSIFICATION_CLASS_FRACTIONS = {
    "Forest": 0.2823712948517941,
    "Shrubland": 0.05499219968798752,
    "Grassland": 0.06747269890795632,
    "Wetlands": 0.008190327613104524,
    "Croplands": 0.22776911076443057,
    "Urban_Built-up": 0.11017940717628705,
    "Barren": 0.01735569422776911,
    "Water": 0.23166926677067082,
}


def clean_string_outputs(string: str) -> str:
    return (
        string.replace("test/", "")
        .replace("jaccardindexadapter_", "")
        .replace("multilabelf1score", "F1")
        .replace("-ewc-segmentation", "")
        .replace("-ewc-classification", "")
        .replace("-ewc-regression", "")
        .replace("-eval", "")
        .replace("s2_era5", "S-2 + Era5 Data")
        .replace("s2_rgb_nir", "S-2")
        .replace("s2_rgb", "S-2 (only RGB)")
        .replace("s2_seasons", "S-2 + Seasons")
        .replace("s2_s1", "S-2 + S-1")
        .replace("s2_climate_zones", "S-2 + Climate Zones")
        .replace("s1_s2_conditional_last", "S-2 + S-1 + Cond. (CM)")
        .replace("s1_s2_unconditional", "S-2 + S-1 (CM)")
        .replace("dfc_2020_feature_extractor-segmentation", "S-2 + S-1")
    )


def make_string_latex_compatible(string: str) -> str:
    return string.replace("_", "\\_")


def safe_string(string, output_file_path: Path):
    with output_file_path.open(mode="w") as output_file:
        output_file.write(string)


def save_plot(output_file_path: Path):
    plt.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.clf()


def uniform_mean(metrics_table: pd.DataFrame) -> pd.Series:
    return metrics_table.mean()


def mse_mean(metrics_table: pd.DataFrame) -> pd.Series:
    return metrics_table.loc["test/mean_squared_error"]


def mae_mean(metrics_table: pd.DataFrame) -> pd.Series:
    return metrics_table.loc["test/mean_absolute_error"]


EXPERIMENTS_DICT = {
    # SEGMENTATION
    "segmentation_w_conditional": {
        "wandb_project_ids": [
            "ssl-diffusion/rs-ddpm-ms-segmentation-egypt-eval",
            "ssl-diffusion/rs-ddpm-ms-segmentation-conditional-eval",
        ],
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-segmentation.yaml",
        "class_metrics": [
            "test/jaccardindexadapter_Herbaceous wetland",
            "test/jaccardindexadapter_Bare",
            "test/jaccardindexadapter_Tree cover",
            # 'test/jaccardindexadapter_Moss and lichen',
            "test/jaccardindexadapter_Shrubland",
            "test/jaccardindexadapter_Cropland",
            "test/jaccardindexadapter_Built-up",
            # 'test/jaccardindexadapter_Snow and Ice',
            "test/jaccardindexadapter_Grassland",
            "test/jaccardindexadapter_Permanenet water bodies",
            # 'test/jaccardindexadapter_Mangroves'
        ],
        "highlight_mode": "max",
        "additional_experiment_names": [
            "s1_s2_conditional_last-ewc-segmentation",
            "s1_s2_unconditional-ewc-segmentation",
        ],
        "exclude_experiment_names": [
            "s2_glo_30_dem-ewc-segmentation",
        ],
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean,
                class_fractions=TEST_SET_SEGMENTATION_CLASS_FRACTIONS,
            ),
        },
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
        "metrics_table_index_name": "mIoU $\\uparrow$",
        "class_fractions": TEST_SET_SEGMENTATION_CLASS_FRACTIONS,
        "run_name_suffix": "-eval",
    },
    "dfc_2020_segmentation": {
        "wandb_project_ids": ["ssl-diffusion/rs-ddpm-ms-dfc-2020"],
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/dfc_2020/segmentation/segmentation.yaml",
        "class_metrics": [
            "val/jaccardindexadapter_Urban_Built-up",
            "val/jaccardindexadapter_Barren",
            "val/jaccardindexadapter_Wetlands",
            "val/jaccardindexadapter_Shrubland",
            "val/jaccardindexadapter_Croplands",
            "val/jaccardindexadapter_Forest",
            "val/jaccardindexadapter_Grassland",
            "val/jaccardindexadapter_Water",
        ],
        "highlight_mode": "max",
        "additional_experiment_names": [
            "[Wrong Splits] dfc_2020_feature_extractor-segmentation",
        ],
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean,
                class_fractions=DFC_2020_TEST_SET_SEGMENTATION_CLASS_FRACTIONS,
            ),
        },
        "metrics_table_index_name": "mIoU $\\uparrow$",
        "class_fractions": DFC_2020_TEST_SET_SEGMENTATION_CLASS_FRACTIONS,
        "transpose_metrics_table": False,
        "label_fraction_value_index": 19,
        "highlight_axis": 1,
    },
    # CLASSIFICATION
    "classification_w_conditional": {
        "wandb_project_ids": [
            "ssl-diffusion/rs-ddpm-ms-classification-gallen-eval",
            "ssl-diffusion/rs-ddpm-ms-classification-france-eval",
        ],
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/dfc_2020/classification/classification.yaml",
        "class_metrics": [
            "test/multilabelf1score_Herbaceous wetland",
            "test/multilabelf1score_Bare",
            "test/multilabelf1score_Tree cover",
            # 'test/multilabelf1score_Moss and lichen',
            "test/multilabelf1score_Shrubland",
            "test/multilabelf1score_Cropland",
            "test/multilabelf1score_Built-up",
            # 'test/multilabelf1score_Snow and Ice',
            "test/multilabelf1score_Grassland",
            "test/multilabelf1score_Permanenet water bodies",
            # 'test/multilabelf1score_Mangroves'
        ],
        "highlight_mode": "max",
        "additional_experiment_names": [
            "s1_s2_conditional_last-ewc-classification",
            "s1_s2_unconditional-ewc-classification",
        ],
        "exclude_experiment_names": [
            "s2_glo_30_dem-ewc-classification",
        ],
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean,
                class_fractions=TEST_SET_CLASSIFICATION_CLASS_FRACTIONS,
            ),
        },
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
        "metrics_table_index_name": "F1 Score $\\uparrow$",
        "class_fractions": TEST_SET_CLASSIFICATION_CLASS_FRACTIONS,
        "run_name_suffix": "-eval",
    },
    "dfc_2020_classification": {
        "wandb_project_ids": ["ssl-diffusion/rs-ddpm-ms-dfc-2020-classification"],
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/dfc_2020/regression/regression.yaml",
        "class_metrics": [
            "val/multilabelf1score_Urban_Built-up",
            "val/multilabelf1score_Barren",
            "val/multilabelf1score_Wetlands",
            "val/multilabelf1score_Shrubland",
            "val/multilabelf1score_Croplands",
            "val/multilabelf1score_Forest",
            "val/multilabelf1score_Grassland",
            "val/multilabelf1score_Water",
        ],
        "highlight_mode": "max",
        "additional_experiment_names": [
            "dfc_2020_feature_extractor-classification",
        ],
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean,
                class_fractions=DFC_2020_TEST_SET_CLASSIFICATION_CLASS_FRACTIONS,
            ),
        },
        "metrics_table_index_name": "F1 Score $\\uparrow$",
        "class_fractions": DFC_2020_TEST_SET_CLASSIFICATION_CLASS_FRACTIONS,
        "transpose_metrics_table": False,
        "label_fraction_value_index": 19,
        "highlight_axis": 1,
    },
    # REGRESSION
    "regression_w_conditional": {
        "wandb_project_ids": [
            "ssl-diffusion/rs-ddpm-ms-regression-egypt-eval",
        ],
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-regression.yaml",
        "class_metrics": [
            "test/mean_squared_error",
            "test/mean_absolute_error",
        ],
        "highlight_mode": "min",
        "additional_experiment_names": [
            "s1_s2_conditional_last-ewc-regression",
            "s1_s2_unconditional-ewc-regression",
        ],
        "exclude_experiment_names": [
            "s2_glo_30_dem-ewc-regression",
        ],
        "averages": {
            "Mean Squared Error $\\downarrow$": mse_mean,
            "Mean Absolute Error $\\downarrow$": mae_mean,
        },
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
        "run_name_suffix": "-eval",
    },
    "dfc_2020_regression": {
        "wandb_project_ids": ["ssl-diffusion/rs-ddpm-ms-regression-egypt"],
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-regression.yaml",
        "class_metrics": [
            "val/mean_squared_error",
            "val/mean_absolute_error",
        ],
        "highlight_mode": "min",
        "additional_experiment_names": [
            "dfc_2020_feature_extractor-regression",
        ],
        "averages": {
            "Mean Squared Error $\\downarrow$": mse_mean,
            "Mean Absolute Error $\\downarrow$": mae_mean,
        },
        "transpose_metrics_table": False,
        "label_fraction_value_index": 19,
        "highlight_axis": 1,
    },
}


def create_report(
    base_output_dir_path,
    wandb_project_ids: str,
    downstream_head_config_path: str,
    class_metrics: List[str],
    highlight_mode: Literal["min", "max"],
    feature_extractor_files: Optional[str] = None,
    averages: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    additional_experiment_names: Optional[List[str]] = None,
    exclude_experiment_names: Optional[List[str]] = None,
    metrics_table_index_name: Optional[str] = None,
    class_fractions: Optional[Dict[str, float]] = None,
    run_name_suffix: Optional[str] = "",
    transpose_metrics_table: bool = False,
    label_fraction_value_index: int = 0,
    highlight_axis: int = 1,
):
    if averages is None:
        averages = {}
    base_output_dir_path = Path(base_output_dir_path)
    # Get file names
    if feature_extractor_files is not None:
        feature_extractor_files = Path(feature_extractor_files)
        feature_extractor_names = list(feature_extractor_files.glob("*.yaml"))
    else:
        feature_extractor_names = []
        assert (
            additional_experiment_names is not None
        ), "Feature extractor files were none but no additional experiment names were provided"
    downstream_head_config_path = Path(downstream_head_config_path)

    EXPERIMENT_NAMES = [
        create_wandb_run_name(
            backbone_name=backbone_path.name,
            downstream_head_name=downstream_head_config_path.name,
        )
        for backbone_path in feature_extractor_names
    ]
    if additional_experiment_names is not None:
        EXPERIMENT_NAMES += additional_experiment_names
    if exclude_experiment_names is not None:
        for e_name in exclude_experiment_names:
            EXPERIMENT_NAMES.remove(e_name)
    LABEL_FRACTION_EXPERIMENT_NAMES = [
        name + f"-lf-{fraction}"
        for name in EXPERIMENT_NAMES
        for fraction in LABEL_FRACTION_PATHS.keys()
    ]
    EVAL_RUN_NAMES = EXPERIMENT_NAMES
    LF_EVAL_RUN_NAMES = LABEL_FRACTION_EXPERIMENT_NAMES
    if run_name_suffix is not None:
        EVAL_RUN_NAMES = [name + run_name_suffix for name in EXPERIMENT_NAMES]
        LF_EVAL_RUN_NAMES = [
            name + run_name_suffix for name in LABEL_FRACTION_EXPERIMENT_NAMES
        ]
    EVAL_RUN_FILTER = {
        "$and": [
            {"display_name": {"$in": EVAL_RUN_NAMES}},
            {"state": {"$eq": "finished"}},
        ]
    }
    LF_EVAL_RUN_FILTER = {
        "$and": [
            {"display_name": {"$in": LF_EVAL_RUN_NAMES}},
            {"state": {"$eq": "finished"}},
        ]
    }

    # Create table showing class-wise overview of metrics
    metrics_table = get_metrics_table(
        wandb_project_ids=wandb_project_ids,
        run_filter=EVAL_RUN_FILTER,
        run_names=EVAL_RUN_NAMES,
        metric_names=class_metrics,
        value_index=label_fraction_value_index,
        value_multiplier=100,
        verbose=True,
    )
    average_results = {}
    for average_name, average_function in averages.items():
        average_results[average_name] = average_function(metrics_table)
    for average_name, average_result in average_results.items():
        metrics_table.loc[average_name] = average_result

    # Store as Latex string
    latex_metrics_table_output_file_path = (
        base_output_dir_path / "class_wise_metrics_table.tex"
    )
    safe_string(
        make_string_latex_compatible(
            clean_string_outputs(
                metrics_table_to_latex(
                    metrics_table,
                    clines="all;data",
                    mode=highlight_mode,
                    index_name=metrics_table_index_name,
                    class_fractions=class_fractions,
                    position=None,
                    transpose=transpose_metrics_table,
                    highlight_axis=highlight_axis,
                )
            )
        ),
        output_file_path=latex_metrics_table_output_file_path,
    )

    # Visualize average of the main metric
    for average in averages.keys():
        main_metric_figure_output_file_path = (
            base_output_dir_path / f"{average}_metric_figure.png"
        )
        visualize_single_metrics_table_metric(
            metrics_table,
            metrics_name=average,
            xlabel_transform=clean_string_outputs,
        )
        save_plot(main_metric_figure_output_file_path)

    # Create label fraction overview
    # Get Data
    lf_metrics_table = get_metrics_table(
        wandb_project_ids=wandb_project_ids,
        run_filter=LF_EVAL_RUN_FILTER,
        run_names=LF_EVAL_RUN_NAMES,
        metric_names=class_metrics,
        value_index=0,
        value_multiplier=100,
    )
    for average_name, average_function in averages.items():
        average_results[average_name] = average_function(lf_metrics_table)
    for average_name, average_result in average_results.items():
        lf_metrics_table.loc[average_name] = average_result

    # Create figure
    for average in averages.keys():
        label_fraction_figure_output_file_path = (
            base_output_dir_path / f"label_fraction_{average}_metric_figure.png"
        )
        get_label_fraction_figure(
            lf_metrics_table=lf_metrics_table,
            metric_key=average,
            experiment_names=EXPERIMENT_NAMES,
            label_fractions=[0.01, 0.1, 0.5],
            name_suffix=run_name_suffix,
            label_transform=clean_string_outputs,
            all_label_values=metrics_table,
        )
        save_plot(label_fraction_figure_output_file_path)


if __name__ == "__main__":
    set_matplotlib_style()

    for name, config in EXPERIMENTS_DICT.items():
        base_output_dir_path = Path(
            f"../../../latex/master_thesis_latex/resources/reports/{name}"
        )
        base_output_dir_path.mkdir(exist_ok=True, parents=True)
        create_report(
            base_output_dir_path=base_output_dir_path,
            **config,
        )
