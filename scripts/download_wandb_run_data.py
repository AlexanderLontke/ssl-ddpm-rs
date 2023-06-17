import os
from remote_sensing_ddpm.evaluation.data_handling.wandb_data import (
    get_wandb_run_histories,
    convert_run_histories_to_df,
)
from remote_sensing_ddpm.evaluation.data_handling.constants import (
    CLASSIFICATION_METRICS,
    SEGMENTATION_METRICS,
    Resolution,
)

from lit_diffusion.diffusion_base.constants import (
    LOGGING_TRAIN_PREFIX,
    LOGGING_VAL_PREFIX,
)

# WandB
WAND_SEGMENTATION_URL = "ssl-diffusion/rs-ddpm-ms-segmentation"
WAND_CLASSIFICATION_URL = "ssl-diffusion/rs-ddpm-ms-classification"
# TODO automate this!
WANDB_EWC_CLASSIFICATION_RUNS = [
    "o1tldx99",
    "5th69f6u",
    "4pavkimp",
    "8sgvkhxz",
    "mlmr0rw1",
    "jh8d6lwr",
    "rjl8mbla",
]

WANDB_EWC_SEGMENTATION_RUNS = [
    "bzjkc6zw",
    "9is20jr4",
    "sigculiy",
    "eylrd8ig",
    "rzd52k27",
    "kiqztjno",
    "wj3o9oho",
]

DOWNSTREAM_TASKS_CONFIG_DIR = "../config/model_configs/downstream_tasks/"
backbone_names = [
    x.split(".")[0]
    for x in os.listdir(os.path.join(DOWNSTREAM_TASKS_CONFIG_DIR, "feature_extractors"))
]
downstream_task_names = [
    file.split(".")[0]
    for file in os.listdir(DOWNSTREAM_TASKS_CONFIG_DIR)
    if os.path.isfile(os.path.join(DOWNSTREAM_TASKS_CONFIG_DIR, file))
]

if __name__ == "__main__":
    # run_histories = get_wandb_run_histories(
    #     project_id=WAND_CLASSIFICATION_URL, run_ids=WANDB_EWC_CLASSIFICATION_RUNS
    # )
    #
    # convert_run_histories_to_df(
    #     metrics=CLASSIFICATION_METRICS,
    #     aggregations=["max"],
    #     split_prefixes=[LOGGING_TRAIN_PREFIX, LOGGING_VAL_PREFIX],
    #     resolution=Resolution.EPOCH,
    #     runs=run_histories,
    #     save_path="./ewc_classification_run_histories.csv",
    # )

    run_histories = get_wandb_run_histories(
        project_id=WAND_SEGMENTATION_URL, run_ids=WANDB_EWC_SEGMENTATION_RUNS,
    )

    convert_run_histories_to_df(
        metrics=SEGMENTATION_METRICS,
        aggregations=["max"],
        split_prefixes=[LOGGING_TRAIN_PREFIX, LOGGING_VAL_PREFIX],
        resolution=Resolution.EPOCH,
        runs=run_histories,
        save_path="./ewc_segmentation_run_histories.csv",
    )
