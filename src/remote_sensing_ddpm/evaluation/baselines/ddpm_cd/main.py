import torch
import data as Data
import remote_sensing_ddpm.evaluation.baselines.ddpm_cd as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from misc.print_diffuse_feats import print_feats
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score

from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.util import set_option_from_sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/ddpm_cd.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        choices=["train", "test"],
        help="Run either train(training + validation) or testing",
        default="train",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_eval", action="store_true")

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Logging
    Logger.setup_logger(
        None, opt["path"]["log"], "train", level=logging.INFO, screen=True
    )
    Logger.setup_logger("test", opt["path"]["log"], "test", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"])

    # Initialize WandbLogger
    if opt["enable_wandb"]:
        import wandb

        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # # Training log
        # wandb.define_metric("epoch")
        # wandb.define_metric("training/train_step")
        # wandb.define_metric("training/*", step_metric="train_step")
        # # Validation log
        # wandb.define_metric("validation/val_step")
        # wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # GPU support
    if opt["gpu_ids"]:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        logger.info("Cudnn enabled")

    # Score aggregation
    score_aggregation = "macro"

    # Set options from sweep
    if opt["phase"] == "train":
        opt["train"]["optimizer"]["lr"] = set_option_from_sweep(
            wandb_config_key="lr", option_value=opt["train"]["optimizer"]["lr"]
        )
    opt["classification_model"]["use_diffusion"] = set_option_from_sweep(
        wandb_config_key="use_diffusion",
        option_value=opt["classification_model"]["use_diffusion"],
    )

    # Loading change-detction datasets.
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train" and args.phase != "test":
            print(f"Creating [train] change-detection dataloader.")
            train_set = Data.create_uc_merced_classification_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            opt["len_train_dataloader"] = len(train_loader)

        elif phase == "val" and args.phase != "test":
            print("Creating [val] change-detection dataloader.")
            val_set = Data.create_uc_merced_classification_dataset(dataset_opt, phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt["len_val_dataloader"] = len(val_loader)

        elif phase == "test" and args.phase == "test":
            print("Creating [test] change-detection dataloader.")
            test_set = Data.create_uc_merced_classification_dataset(dataset_opt, phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt["len_test_dataloader"] = len(test_loader)

    logger.info("Initial Dataset Finished")

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info("Initial Diffusion Model Finished")

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"][opt["phase"]], schedule_phase=opt["phase"]
    )

    # Creating classification model
    classifier = Model.create_classification_model(opt)

    #################
    # Training loop #
    #################
    n_epoch = opt["train"]["n_epoch"]
    best_acc = 0.0
    start_epoch = 0
    if opt["phase"] == "train":
        for current_epoch in range(start_epoch, n_epoch):
            train_result_path = "{}/train/{}".format(
                opt["path"]["results"], current_epoch
            )
            os.makedirs(train_result_path, exist_ok=True)

            ################
            #   buffers    #
            ################
            labels = []
            predictions = []
            epoch_loss = []

            ################
            #   training   #
            ################
            message = (
                "lr: %0.7f\n" % classifier.opt_classification.param_groups[0]["lr"]
            )
            logger.info(message)
            with tqdm(
                enumerate(train_loader),
                desc="Training Network",
                total=len(train_loader),
            ) as iterator:
                for current_step, train_data in iterator:
                    # Feeding data to diffusion model and get features
                    diffusion.feed_data(train_data)
                    feats = []
                    for t in opt["classification_model"]["time_steps"]:
                        fe, fd = diffusion.get_single_representation(
                            t=t
                        )  # np.random.randint(low=2, high=8)
                        if opt["classification_model"]["feat_type"] == "dec":
                            feats.append(fd)
                            # Uncomment the following lines to visualize features from the diffusion model
                            # for level in range(0, len(fd_A_t)):
                            #     print_feats(
                            #         opt=opt,
                            #         train_data=train_data,
                            #         feats_A=fd_A_t,
                            #         feats_B=fd_B_t,
                            #         level=level,
                            #         t=t,
                            #     )
                            del fe
                        else:
                            feats.append(fe)
                            del fd

                    # for i in range(0, len(fd_A)):
                    #     print(fd_A[i].shape)

                    # Feeding features from the diffusion model to the CD model
                    classifier.feed_data(feats, train_data)
                    current_predictions, current_loss = classifier.optimize_parameters()
                    labels.extend(train_data["L"].cpu().numpy())
                    predictions.extend(
                        torch.argmax(current_predictions, dim=1).cpu().numpy()
                    )
                    epoch_loss.append(current_loss)
                    if wandb_logger:
                        wandb_logger._wandb.log(
                            {
                                "training/iter_loss": current_loss,
                            }
                        )

                    # log running batch status
                    message = f"[Training CD]. epoch: [{current_epoch}/{n_epoch-1}]. Loss: {current_loss}"
                    iterator.desc = message

                if wandb_logger:
                    wandb_logger._wandb.log(
                        {
                            "training/epoch_accuracy": accuracy_score(
                                y_true=labels,
                                y_pred=predictions,
                            ),
                            "training/epoch_recall": recall_score(
                                y_true=labels,
                                y_pred=predictions,
                                average=score_aggregation,
                            ),
                            "training/epoch_precision": precision_score(
                                y_true=labels,
                                y_pred=predictions,
                                average=score_aggregation,
                            ),
                            "training/epoch_loss": np.mean(epoch_loss),
                        }
                    )
                classifier._update_lr_schedulers()

                ##################
                ### validation ###
                ##################
                if current_epoch % opt["train"]["val_freq"] == 0:
                    val_result_path = "{}/val/{}".format(
                        opt["path"]["results"], current_epoch
                    )
                    os.makedirs(val_result_path, exist_ok=True)
                    val_predictions = []
                    val_labels = []
                    val_loss = []
                    with tqdm(enumerate(val_loader), total=len(val_loader)) as val_iter:
                        for current_step, val_data in val_iter:
                            # Feed data to diffusion model
                            diffusion.feed_data(val_data)
                            feats = []
                            for t in opt["classification_model"]["time_steps"]:
                                fe, fd = diffusion.get_single_representation(
                                    t=t
                                )  # np.random.randint(low=2, high=8)
                                if opt["classification_model"]["feat_type"] == "dec":
                                    feats.append(fd)
                                    del fe
                                else:
                                    feats.append(fe)
                                    del fd

                            # Feed data to CD model
                            classifier.feed_data(feats, val_data)
                            current_predictions, current_loss = classifier.test()
                            val_labels.extend(val_data["L"].cpu().numpy())
                            val_predictions.extend(
                                torch.argmax(current_predictions, dim=1).cpu().numpy()
                            )
                            val_loss.append(current_loss)
                            message = f"[Validating classifier]. Loss: {current_loss}"
                            val_iter.desc = message

                            # log running batch status for val data
                            if wandb_logger:
                                wandb_logger._wandb.log(
                                    {"validation/iter_loss": current_loss}
                                )
                    validation_accuracy = accuracy_score(
                        y_true=val_labels,
                        y_pred=val_predictions,
                    )
                    if wandb_logger:
                        wandb_logger._wandb.log(
                            {
                                "validation/epoch_accuracy": validation_accuracy,
                                "validation/epoch_recall": recall_score(
                                    y_true=val_labels,
                                    y_pred=val_predictions,
                                    average=score_aggregation,
                                ),
                                "validation/epoch_precision": precision_score(
                                    y_true=val_labels,
                                    y_pred=val_predictions,
                                    average=score_aggregation,
                                ),
                                "validation/epoch_loss": np.mean(val_loss),
                            }
                        )

                    if validation_accuracy > best_acc:
                        is_best_model = True
                        best_acc = validation_accuracy
                        logger.info(
                            "[Validation CD] Best model updated. Saving the models (current + best) and training states."
                        )
                    else:
                        is_best_model = False
                        logger.info(
                            "[Validation CD]Saving the current cd model and training states."
                        )
                    logger.info("--- Proceed To The Next Epoch ----\n \n")

                    classifier.save_network(current_epoch, is_best_model=is_best_model)

                if wandb_logger:
                    wandb_logger.log_metrics({"epoch": current_epoch - 1})
            logger.info("End of training.")
    else:
        logger.info("Begin Model Evaluation (testing).")
        test_result_path = "{}/test/".format(opt["path"]["results"])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger("test")  # test logger
        testing_predictions = []
        testing_labels = []
        testing_loss = []
        with tqdm(enumerate(test_loader)) as test_iter:
            for current_step, test_data in test_iter:
                # Feed data to diffusion model
                diffusion.feed_data(test_data)
                feats = []
                for t in opt["classification_model"]["time_steps"]:
                    fe, fd = diffusion.get_single_representation(
                        t=t
                    )  # np.random.randint(low=2, high=8)
                    if opt["classification_model"]["feat_type"] == "dec":
                        feats.append(fd)
                        del fe
                    else:
                        feats.append(fe)
                        del fd

                # Feed data to CD model
                classifier.feed_data(feats, test_data)
                current_predictions, current_loss = classifier.test()
                testing_labels.extend(test_data["L"].cpu().numpy())
                testing_predictions.extend(
                    torch.argmax(current_predictions, dim=1).cpu().numpy()
                )
                testing_loss.append(current_loss)
                message = f"[Testing classifier]. Loss: {current_loss}"
                test_iter.desc = message
                if wandb_logger:
                    wandb_logger._wandb.log({"testing/iter_loss": current_loss})

        if wandb_logger:
            wandb_logger._wandb.log_metrics(
                {
                    "validation/epoch_accuracy": accuracy_score(
                        y_true=testing_labels,
                        y_pred=testing_predictions,
                    ),
                    "validation/epoch_recall": recall_score(
                        y_true=testing_labels,
                        y_pred=testing_predictions,
                        average=score_aggregation,
                    ),
                    "validation/epoch_precision": precision_score(
                        y_true=testing_labels,
                        y_pred=testing_predictions,
                        average=score_aggregation,
                    ),
                    "validation/epoch_loss": np.mean(testing_loss),
                }
            )
        logger.info("End of testing...")
