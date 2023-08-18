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

TEST_SET_CLASS_FRACTIONS = {
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


def clean_string_outputs(string: str) -> str:
    return (
        string.replace("test/", "")
        .replace("jaccardindexadapter_", "mIoU ")
        .replace("multilabelf1score", "F1")
        .replace("-ewc-segmentation", "")
        .replace("-ewc-classification", "")
        .replace("-ewc-regression", "")
        .replace("-eval", "")
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


EXPERIMENTS_DICT = {
    "segmentation": {
        "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-segmentation-egypt-eval",
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
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean, class_fractions=TEST_SET_CLASS_FRACTIONS
            ),
        },
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
    },
    "segmentation_w_supervised": {
        "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-segmentation-egypt-eval",
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
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean, class_fractions=TEST_SET_CLASS_FRACTIONS
            ),
        },
        "additional_experiment_names": ["supervised_s1_s2-supervised_ewc_segmentation"],
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
    },
    "conditional_models": {
        "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-segmentation-conditional-eval",
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
        "averages": {
            UNIFORM_AVERAGE_NAME: uniform_mean,
            CLASS_WEIGHTED_AVERAGE_NAME: partial(
                calculate_class_weighted_mean, class_fractions=TEST_SET_CLASS_FRACTIONS
            ),
        },
        "additional_experiment_names": ["supervised_s1_s2_conditional-ewc-segmentation", "s1_s2_conditional_last-ewc-segmentation", "s1_s2_unconditional_last-ewc-segmentation"],
        "feature_extractor_files": None,
    },
    "classification": {
        "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-classification-france-eval",
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-classification.yaml",
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
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
    },
    "regression": {
        "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-regression-egypt-eval",
        "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-regression.yaml",
        "class_metrics": [
            "test/mean_squared_error",
            "test/mean_absolute_error",
        ],
        "highlight_mode": "min",
        "feature_extractor_files": "../../config/model_configs/downstream_tasks/feature_extractors",
    },
}


def create_report(
    base_output_dir_path,
    wandb_project_id: str,
    downstream_head_config_path: str,
    feature_extractor_files: Optional[str],
    class_metrics: List[str],
    highlight_mode: Literal["min", "max"],
    averages: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    additional_experiment_names: Optional[List[str]] = None,
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
        assert additional_experiment_names is not None, "Feature extractor files were none but no additional experiment names were provided"
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
    LABEL_FRACTION_EXPERIMENT_NAMES = [
        name + f"-lf-{fraction}"
        for name in EXPERIMENT_NAMES
        for fraction in LABEL_FRACTION_PATHS.keys()
    ]
    EVAL_RUN_NAMES = [name + "-eval" for name in EXPERIMENT_NAMES]
    LF_EVAL_RUN_NAMES = [name + "-eval" for name in LABEL_FRACTION_EXPERIMENT_NAMES]
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
        wandb_project_id=wandb_project_id,
        run_filter=EVAL_RUN_FILTER,
        run_names=EVAL_RUN_NAMES,
        metric_names=class_metrics,
        value_index=0,
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
        wandb_project_id=wandb_project_id,
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
            name_suffix="-eval",
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
