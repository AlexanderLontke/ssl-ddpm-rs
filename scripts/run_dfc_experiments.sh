#!/bin/bash

# python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/segmentation/dfc_2020_feature_extractor.yaml -d config/model_configs/downstream_tasks/dfc_2020/segmentation/segmentation.yaml -r 5
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/regression/dfc_2020_feature_extractor.yaml -d config/model_configs/downstream_tasks/dfc_2020/regression/regression.yaml -r 5
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/classification/dfc_2020_feature_extractor.yaml -d config/model_configs/downstream_tasks/dfc_2020/classification/classification.yaml -r 5



# Label fractions
# Classification
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/classification/dfc_2020_feature_extractor-lf-1.yaml -d config/model_configs/downstream_tasks/dfc_2020/classification/classification.yaml -r 1
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/classification/dfc_2020_feature_extractor-lf-10.yaml -d config/model_configs/downstream_tasks/dfc_2020/classification/classification.yaml -r 1
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/classification/dfc_2020_feature_extractor-lf-50.yaml -d config/model_configs/downstream_tasks/dfc_2020/classification/classification.yaml -r 1


# # Segmentations
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/segmentation/dfc_2020_feature_extractor-lf-1.yaml -d config/model_configs/downstream_tasks/dfc_2020/segmentation/segmentation.yaml -r 1
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/segmentation/dfc_2020_feature_extractor-lf-10.yaml -d config/model_configs/downstream_tasks/dfc_2020/segmentation/segmentation.yaml -r 1
python src/remote_sensing_ddpm/run_downstream_tasks.py -b config/model_configs/downstream_tasks/dfc_2020/segmentation/dfc_2020_feature_extractor-lf-50.yaml -d config/model_configs/downstream_tasks/dfc_2020/segmentation/segmentation.yaml -r 1