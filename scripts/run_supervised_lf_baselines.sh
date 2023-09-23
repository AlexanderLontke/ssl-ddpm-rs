#!/bin/bash

# Segmentation
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-segmentation-lf-1.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-segmentation-lf-10.yaml
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-segmentation-lf-50.yaml

# Classification
# python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-classification-lf-1.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-classification-lf-10.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-classification-lf-50.yaml

# Regression
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-regression-lf-1.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-regression-lf-10.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-regression-lf-50.yaml

# Full Runs
python -m lit_diffusion.train -c config/baselines/supervised-baselines/segmentation/supervised-dfc2020-segmentation-seed-42.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/classification/supervised-dfc2020-classification-seed-42.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/regression/supervised-dfc2020-regression-seed-42.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/segmentation/supervised-dfc2020-segmentation-seed-43.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/classification/supervised-dfc2020-classification-seed-43.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/regression/supervised-dfc2020-regression-seed-43.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/segmentation/supervised-dfc2020-segmentation-seed-44.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/classification/supervised-dfc2020-classification-seed-44.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/regression/supervised-dfc2020-regression-seed-44.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/segmentation/supervised-dfc2020-segmentation-seed-45.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/classification/supervised-dfc2020-classification-seed-45.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/regression/supervised-dfc2020-regression-seed-45.yaml

python -m lit_diffusion.train -c config/baselines/supervised-baselines/segmentation/supervised-dfc2020-segmentation-seed-46.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/classification/supervised-dfc2020-classification-seed-46.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/regression/supervised-dfc2020-regression-seed-46.yaml