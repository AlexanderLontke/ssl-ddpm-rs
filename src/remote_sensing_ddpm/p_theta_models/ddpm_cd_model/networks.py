import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger("base")
from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.cd_modules.cd_head_v2 import (
    cd_head_v2,
)
from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.classification_modules import (
    ClassificationHead,
)

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="kaiming", scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info("Initialization method [{:s}]".format(init_type))
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [{:s}] not implemented".format(init_type)
        )


####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt["model"]
    if model_opt["which_model_G"] == "ddpm":
        from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.ddpm_modules import (
            diffusion,
        )
        from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.ddpm_modules import unet
    elif model_opt["which_model_G"] == "sr3":
        from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.sr3_modules import unet
        from remote_sensing_ddpm.p_theta_models.ddpm_cd_model.sr3_modules import (
            diffusion,
        )
    if ("norm_groups" not in model_opt["unet"]) or model_opt["unet"][
        "norm_groups"
    ] is None:
        model_opt["unet"]["norm_groups"] = 32
    model = unet.UNet(
        in_channel=model_opt["unet"]["in_channel"],
        out_channel=model_opt["unet"]["out_channel"],
        norm_groups=model_opt["unet"]["norm_groups"],
        inner_channel=model_opt["unet"]["inner_channel"],
        channel_mults=model_opt["unet"]["channel_multiplier"],
        attn_res=model_opt["unet"]["attn_res"],
        res_blocks=model_opt["unet"]["res_blocks"],
        dropout=model_opt["unet"]["dropout"],
        image_size=model_opt["diffusion"]["image_size"],
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt["diffusion"]["image_size"],
        channels=model_opt["diffusion"]["channels"],
        loss_type=model_opt["diffusion"]["loss"],  # L1 or L2
        conditional=model_opt["diffusion"]["conditional"],
        schedule_opt=model_opt["beta_schedule"]["train"],
    )
    # Skip weight initialization if checkpoint is loaded
    if opt["phase"] == "train" and not opt["path"]["resume_state"]:
        # init_weights(netG, init_type='kaiming', scale=0.1)
        logger.info("Initializing diffusion model weights")
        init_weights(netG, init_type="orthogonal")
    if opt["gpu_ids"] and opt["distributed"]:
        assert torch.cuda.is_available()
        print("Distributed training")
        netG = nn.DataParallel(netG)
    return netG


# Change Detection Network
def define_CD(opt):
    cd_model_opt = opt["model_cd"]
    diffusion_model_opt = opt["model"]

    # Define change detection network head
    # netCD = cd_head(feat_scales=cd_model_opt['feat_scales'],
    #                 out_channels=cd_model_opt['out_channels'],
    #                 inner_channel=diffusion_model_opt['unet']['inner_channel'],
    #                 channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
    #                 img_size=cd_model_opt['output_cm_size'],
    #                 psp=cd_model_opt['psp'])
    netCD = cd_head_v2(
        feat_scales=cd_model_opt["feat_scales"],
        out_channels=cd_model_opt["out_channels"],
        inner_channel=diffusion_model_opt["unet"]["inner_channel"],
        channel_multiplier=diffusion_model_opt["unet"]["channel_multiplier"],
        img_size=cd_model_opt["output_cm_size"],
        time_steps=cd_model_opt["t"],
    )

    # Initialize the change detection head if it is 'train' phase
    if opt["phase"] == "train":
        # Try different initialization methods
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netCD, init_type="orthogonal")
    if opt["gpu_ids"] and opt["distributed"]:
        assert torch.cuda.is_available()
        netCD = nn.DataParallel(netCD)

    return netCD


def define_classification(opt):
    classification_model_opt = opt["classification_model"]
    diffusion_model_opt = opt["model"]
    # Define classification head
    classification_net = ClassificationHead(
        **classification_model_opt,
        inner_channel=diffusion_model_opt["unet"]["inner_channel"],
        channel_multiplier=diffusion_model_opt["unet"]["channel_multiplier"],
    )

    # Initialize the change detection head if it is 'train' phase
    if opt["phase"] == "train":
        # Try different initialization methods
        # init_weights(classification_net, init_type='kaiming', scale=0.1)
        init_weights(classification_net, init_type="orthogonal")
    if opt["gpu_ids"] and opt["distributed"]:
        assert torch.cuda.is_available()
        classification_net = nn.DataParallel(classification_net)
    return classification_net
