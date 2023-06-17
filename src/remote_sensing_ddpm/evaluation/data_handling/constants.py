from enum import Enum


class Resolution(Enum):
    EPOCH = "_epoch"
    STEP = "_step"


MINIMUM_AGGREGATION = "min"
MAXIMUM_AGGREGATION = "max"

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1"]
SEGMENTATION_METRICS = ["mIoU", "accuracy"]
