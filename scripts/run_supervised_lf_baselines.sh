#!/bin/bash

# Segmentation
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-segmentation-lf-1.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-segmentation-lf-10.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-segmentation-lf-50.yaml

# Classification
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-classification-lf-1.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-classification-lf-10.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-classification-lf-50.yaml

# Regression
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-regression-lf-1.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-regression-lf-10.yaml
python -m lit_diffusion.train -c config/baselines/supervised-baselines/label-fractions/supervised-dfc2020-regression-lf-50.yaml