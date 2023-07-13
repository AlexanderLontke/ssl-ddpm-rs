# Pytorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# lit diffusion imports
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import PL_WANDB_LOGGER_CONFIG_KEY

# Remote Sensing DDPM imports
from remote_sensing_ddpm.downstream_tasks.lit_downstream_task import LitDownstreamTask


def run_test(checkpoint_path, map_location, test_dataloader_config, wandb_logger, *args, **kwargs):
    # TODO load model
    diffusion_pl_module = LitDownstreamTask.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=map_location,
        downstream_model=None,
        learning_rate=None,
        loss=None,
        target_key=None,
    )
    # TODO load test dataloader
    test_dataloader = instantiate_python_class_from_string_config(
        test_dataloader_config
    )
    # TODO create trainer
    trainer = pl.Trainer(
        model=diffusion_pl_module,
        logger=wandb_logger,
    )
    # TODO run trainer.test()
    trainer.test(dataloaders=test_dataloader)


if __name__ == "__main__":
    # TODO match run name with checkpoint 
    # -> match run name with dataloader
    config = {
        PL_WANDB_LOGGER_CONFIG_KEY: {
            "project": "test-stuff",
        },
        "checkpoint_path": "/netscratch2/alontke/master_thesis/code/ssl-ddpm-rs/rs-ddpm-ms-segmentation-egypt/6f43r1u9/checkpoints/last.ckpt",
        "map_location": "cuda",
        "test_dataloader_config": {
            "module": "remote_sensing_ddpm.utils.ffcv_loader_wrapper.FFCVLoaderWrapper",
            "kwargs": {
                "fname": "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-20-multi-label-ewc-test.beton",
                "batch_size": 24,
                "num_workers": 4,
                "order": {
                    "module": "ffcv.loader.OrderOption",
                    "args": [ 1 ] # 1 is sequential
                },
                "pipelines": {
                    'climate_zone': None, 
                    'elevation_differ': None, 
                    'era_5': None, 
                    'esa_worldcover': None, 
                    'glo_30_dem': None, 
                    'multiclass_numer': None, 
                    'multiclass_one_h': None, 
                    'season_s1': None, 
                    'season_s2': None, 
                    'sentinel_1': None, 
                    'sentinel_2': [{'module': 'ffcv.fields.decoders.NDArrayDecoder'}, {'module': 'remote_sensing_core.transforms.ffcv.channel_selector.ChannelSelector', 'kwargs': {'channels': [3, 2, 1]}}, {'module': 'remote_sensing_core.transforms.ffcv.clipping.Clipping', 'kwargs': {'clip_values': [0, 10000]}}, {'module': 'remote_sensing_core.transforms.ffcv.min_max_scaler.MinMaxScaler', 'kwargs': {'minimum_value': 0, 'maximum_value': 10000}}, {'module': 'remote_sensing_core.transforms.ffcv.padding.Padding', 'kwargs': {'padding': 4, 'padding_value': 0}}, {'module': 'remote_sensing_core.transforms.ffcv.convert.Convert', 'kwargs': {'target_dtype': 'float32'}}, {'module': 'ffcv.transforms.ToTensor'}], 'field_names': None}
            }
        },
    }
    config["wandb_logger"] = WandbLogger(**config[PL_WANDB_LOGGER_CONFIG_KEY], config=config)
    run_test(**config)
