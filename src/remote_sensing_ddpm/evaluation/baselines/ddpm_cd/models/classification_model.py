import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import remote_sensing_ddpm.evaluation.baselines.ddpm_cd.models.networks as networks
from .base_model import BaseModel
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.misc.torchutils import (
    get_scheduler,
)

logger = logging.getLogger("base")
CLASSIFICATION_MODEL_CONFIG_KEY = "classification_model"


class Classfication(BaseModel):
    def __init__(self, opt):
        super(Classfication, self).__init__(opt)
        # define network and load pretrained models
        self.classification_net = self.set_device(networks.define_classification(opt))

        # set loss and load resume state
        self.loss_type = opt[CLASSIFICATION_MODEL_CONFIG_KEY]["loss_type"]
        if self.loss_type == "ce":
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()

        if self.opt["phase"] == "train":
            self.classification_net.train()
            # find the parameters to optimize
            optim_classification_params = list(self.classification_net.parameters())

            if opt["train"]["optimizer"]["type"] == "adam":
                self.opt_classification = torch.optim.Adam(
                    optim_classification_params, lr=opt["train"]["optimizer"]["lr"]
                )
            elif opt["train"]["optimizer"]["type"] == "adamw":
                self.opt_classification = torch.optim.AdamW(
                    optim_classification_params, lr=opt["train"]["optimizer"]["lr"]
                )
            else:
                raise NotImplementedError(
                    "Optimizer [{:s}] not implemented".format(
                        opt["train"]["optimizer"]["type"]
                    )
                )

            # Define learning rate sheduler
            self.exp_lr_scheduler_classification_net = get_scheduler(
                optimizer=self.opt_classification, args=opt["train"]
            )
        else:
            self.classification_net.eval()

        self.load_network()
        self.print_network()

        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]
        self.use_diffusion = opt["classification_model"]["use_diffusion"]

    # Feeding all data to the classification model
    def feed_data(self, feats, data):
        self.feats = feats
        self.data = self.set_device(data)

    # Optimize the parameters of the classification model
    def optimize_parameters(self):
        self.opt_classification.zero_grad()
        if self.use_diffusion:
            self.prediction = self.classification_net(self.feats)
        else:
            self.prediction = self.classification_net(self.data["image"])
        l_classification = self.loss_func(self.prediction, self.data["L"].long())
        l_classification.backward()
        self.opt_classification.step()
        return self.prediction, l_classification.item()

    # Testing on given data
    def test(self):
        self.classification_net.eval()
        with torch.no_grad():
            if isinstance(self.classification_net, nn.DataParallel):
                self.prediction = self.classification_net.module.forward(
                    self.feats
                )
            else:
                self.prediction = self.classification_net(self.feats)
            l_classification = self.loss_func(self.prediction, self.data["L"].long())
        self.classification_net.train()
        return self.prediction, l_classification.item()

    # Printing the classification network
    def print_network(self):
        s, n = self.get_network_description(self.classification_net)
        if isinstance(self.classification_net, nn.DataParallel):
            net_struc_str = "{} - {}".format(
                self.classification_net.__class__.__name__,
                self.classification_net.module.__class__.__name__,
            )
        else:
            net_struc_str = "{}".format(self.classification_net.__class__.__name__)

        logger.info(
            "CLassification Detection Network structure: {}, with parameters: {:,d}".format(
                net_struc_str, n
            )
        )
        logger.info(s)
        return s

    # Saving the network parameters
    def save_network(self, epoch, is_best_model=False):
        classificaton_gen_path = os.path.join(
            self.opt["path"]["checkpoint"],
            "classification_model_E{}_gen.pth".format(epoch),
        )
        classification_opt_path = os.path.join(
            self.opt["path"]["checkpoint"],
            "classification_model_E{}_opt.pth".format(epoch),
        )

        if is_best_model:
            best_classification_gen_path = os.path.join(
                self.opt["path"]["checkpoint"],
                "best_classification_model_gen.pth".format(epoch),
            )
            best_classification_opt_path = os.path.join(
                self.opt["path"]["checkpoint"],
                "best_classification_model_opt.pth".format(epoch),
            )

        # Save classification model pareamters
        network = self.classification_net
        if isinstance(self.classification_net, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, classificaton_gen_path)
        if is_best_model:
            torch.save(state_dict, best_classification_gen_path)

        # Save classification optimizer paramers
        opt_state = {"epoch": epoch, "scheduler": None, "optimizer": None}
        opt_state["optimizer"] = self.opt_classification.state_dict()
        torch.save(opt_state, classification_opt_path)
        if is_best_model:
            torch.save(opt_state, best_classification_opt_path)

        # Print info
        logger.info(
            "Saved current classification model in [{:s}] ...".format(
                classificaton_gen_path
            )
        )
        if is_best_model:
            logger.info(
                "Saved best classification model in [{:s}] ...".format(
                    best_classification_gen_path
                )
            )

    # Loading pre-trained classification network
    def load_network(self):
        load_path = self.opt["path_classification"]["resume_state"]
        if load_path is not None:
            logger.info(
                "Loading pretrained model for Classification model [{:s}] ...".format(
                    load_path
                )
            )
            gen_path = "{}_gen.pth".format(load_path)
            opt_path = "{}_opt.pth".format(load_path)

            # change detection model
            network = self.classification_net
            if isinstance(self.classification_net, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=True)

            if self.opt["phase"] == "train":
                opt = torch.load(opt_path)
                self.opt_classification.load_state_dict(opt["optimizer"])
                self.begin_step = opt["iter"]
                self.begin_epoch = opt["epoch"]

    # Functions related to learning rate scheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_classification_net.step()
