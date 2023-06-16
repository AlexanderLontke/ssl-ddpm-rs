from enum import Enum


class Resolution(Enum):
    EPOCH = "_epoch"
    STEP = "_step"


MINIMUM_AGGREGATION = "min"
MAXIMUM_AGGREGATION = "max"

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1"]

# WandB
WAND_CLASSIFICATION_URL = "ssl-diffusion/rs-ddpm-ms-classification"
WANDB_EWC_CLASSIFICATION_RUNS = [
    "o1tldx99",
    "5th69f6u",
    "4pavkimp",
    "8sgvkhxz",
    "mlmr0rw1",
    "jh8d6lwr",
    "rjl8mbla",
]
