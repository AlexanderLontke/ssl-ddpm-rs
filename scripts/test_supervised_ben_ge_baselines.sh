#!/bin/bash

TEST_BETON_FILE="/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-test.beton"

# Label Fractions
# Classification
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-classification-lf-1.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-classification-lf-10.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-classification-lf-50.yaml -t $TEST_BETON_FILE

# Regression
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-regression-lf-1.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-regression-lf-10.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-regression-lf-50.yaml -t $TEST_BETON_FILE

# Segmentation
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-segmentation-lf-1.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-segmentation-lf-10.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/label-fractions/supervised-ben-ge-segmentation-lf-50.yaml -t $TEST_BETON_FILE

# Full runs
python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/classification/supervised-ben-ge-classification-seed-42.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/regression/supervised-ben-ge-regression-seed-42.yaml -t $TEST_BETON_FILE
# python -m lit_diffusion.test -c config/baselines/supervised-baselines/ben-ge/segmentation/supervised-ben-ge_segmentation-seed-42.yaml -t $TEST_BETON_FILE
