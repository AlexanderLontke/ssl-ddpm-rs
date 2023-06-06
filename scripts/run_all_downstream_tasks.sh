#!/bin/bash
BACKBONE_CONFIG=$1

# Run all six downstream tasks one after the other
python src/remote_sensing_ddpm/train_downstream_tasks.py -b $BACKBONE_CONFIG -d config/model_configs/downstream_tasks/climate-zone-classification.yaml
python src/remote_sensing_ddpm/train_downstream_tasks.py -b $BACKBONE_CONFIG -d config/model_configs/downstream_tasks/elevation-regression.yaml
python src/remote_sensing_ddpm/train_downstream_tasks.py -b $BACKBONE_CONFIG -d config/model_configs/downstream_tasks/elevation-segmentation.yaml
python src/remote_sensing_ddpm/train_downstream_tasks.py -b $BACKBONE_CONFIG -d config/model_configs/downstream_tasks/ewc-classification.yaml
python src/remote_sensing_ddpm/train_downstream_tasks.py -b $BACKBONE_CONFIG -d config/model_configs/downstream_tasks/ewc-regression.yaml
python src/remote_sensing_ddpm/train_downstream_tasks.py -b $BACKBONE_CONFIG -d config/model_configs/downstream_tasks/ewc-segmentation.yaml