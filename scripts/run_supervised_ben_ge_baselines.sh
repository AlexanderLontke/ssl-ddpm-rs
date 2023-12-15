#!/bin/bash

# Label Fractions
# Classification
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-classification-lf-1.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-classification-lf-10.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-classification-lf-50.yaml

# Regression
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-regression-lf-1.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-regression-lf-10.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-regression-lf-50.yaml

# Segmentation
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-segmentation-lf-1.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-segmentation-lf-10.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-segmentation-lf-50.yaml

# Full runs
python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/classification/supervised-ben-ge-classification-seed-42.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/regression/supervised-ben-ge-regression-seed-42.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/segmentation/supervised-ben-ge_segmentation-seed-42.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/classification/supervised-ben-ge-classification-seed-43.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/regression/supervised-ben-ge-regression-seed-43.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/segmentation/supervised-ben-ge_segmentation-seed-43.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/classification/supervised-ben-ge-classification-seed-44.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/regression/supervised-ben-ge-regression-seed-44.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/segmentation/supervised-ben-ge_segmentation-seed-44.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/classification/supervised-ben-ge-classification-seed-45.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/regression/supervised-ben-ge-regression-seed-45.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/segmentation/supervised-ben-ge_segmentation-seed-45.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/classification/supervised-ben-ge-classification-seed-46.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/regression/supervised-ben-ge-regression-seed-46.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/ben-ge/segmentation/supervised-ben-ge_segmentation-seed-46.yaml