from typing import List
from pathlib import Path

# Matplotlib
import matplotlib.pyplot as plt

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

UNIFORM_AVERAGE_NAME = "Uniform Average"


def clean_string_outputs(string: str) -> str:
    return (
        string.replace("test/", "")
        .replace("jaccardindexadapter_", "mIoU ")
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
    },
    # "classification": {
    #     "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-classification-france-eval",
    #     "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-classification.yaml",
    # },
    # "regression": {
    #     "wandb_project_id": "ssl-diffusion/rs-ddpm-ms-regression-egypt-eval",
    #     "downstream_head_config_path": "../../config/model_configs/downstream_tasks/tier_1/ewc-regression.yaml",
    # },
}


def create_report(
    base_output_dir_path,
    wandb_project_id: str,
    downstream_head_config_path: str,
    feature_extractor_files: str,
    class_metrics: List[str],
):
    base_output_dir_path = Path(base_output_dir_path)
    # Get file names
    feature_extractor_files = Path(feature_extractor_files)
    FEATURE_EXTRACTOR_NAMES = list(feature_extractor_files.glob("*.yaml"))
    downstream_head_config_path = Path(downstream_head_config_path)

    EXPERIMENT_NAMES = [
        create_wandb_run_name(
            backbone_name=backbone_path.name,
            downstream_head_name=downstream_head_config_path.name,
        )
        for backbone_path in FEATURE_EXTRACTOR_NAMES
    ]  # + ["supervised_s1_s2-supervised_ewc_segmentation"]
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
    metrics_table.loc[UNIFORM_AVERAGE_NAME] = metrics_table.mean()

    # Store as Latex string
    latex_metrics_table_output_file_path = (
        base_output_dir_path / "class_wise_metrics_table.tex"
    )
    safe_string(
        make_string_latex_compatible(
            clean_string_outputs(
                metrics_table_to_latex(metrics_table, clines="all;data")
            )
        ),
        output_file_path=latex_metrics_table_output_file_path,
    )

    # Visualize average of the main metric
    main_metric_figure_output_file_path = (
        base_output_dir_path / f"{UNIFORM_AVERAGE_NAME}_metric_figure.png"
    )
    visualize_single_metrics_table_metric(
        metrics_table,
        metrics_name=UNIFORM_AVERAGE_NAME,
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
    lf_metrics_table.loc[UNIFORM_AVERAGE_NAME] = lf_metrics_table.mean()

    # Create figure
    label_fraction_figure_output_file_path = (
        base_output_dir_path
        / f"label_fraction_{UNIFORM_AVERAGE_NAME}_metric_figure.png"
    )
    get_label_fraction_figure(
        lf_metrics_table=lf_metrics_table,
        metric_key=UNIFORM_AVERAGE_NAME,
        experiment_names=EXPERIMENT_NAMES,
        label_fractions=[0.01, 0.1, 0.5],
        name_suffix="-eval",
        label_transform=clean_string_outputs,
    )
    save_plot(label_fraction_figure_output_file_path)


if __name__ == "__main__":
    set_matplotlib_style()

    for name, config in EXPERIMENTS_DICT.items():
        base_output_dir_path = Path(f"./reports/{name}")
        base_output_dir_path.mkdir(exist_ok=True, parents=True)
        create_report(
            base_output_dir_path=base_output_dir_path,
            feature_extractor_files="../../config/model_configs/downstream_tasks/feature_extractors",
            **config,
        )
