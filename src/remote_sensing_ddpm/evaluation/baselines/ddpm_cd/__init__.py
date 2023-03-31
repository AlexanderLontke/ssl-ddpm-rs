"""
Baseline: "DDPM-CD: Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models"
https://arxiv.org/abs/2206.11892
Taken from: https://github.com/wgcban/ddpm-cd/tree/master/model
"""
import logging

logger = logging.getLogger("base")


def create_model(opt):
    from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.model import DDPM as M

    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m


def create_CD_model(opt):
    from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.cd_model import CD as M

    m = M(opt)
    logger.info("Cd Model [{:s}] is created.".format(m.__class__.__name__))
    return m


def create_classification_model(opt):
    from remote_sensing_ddpm.p_theta_models.ddpm_cd_model import (
        Classfication as M,
    )

    m = M(opt)
    logger.info("Cd Model [{:s}] is created.".format(m.__class__.__name__))
    return m
