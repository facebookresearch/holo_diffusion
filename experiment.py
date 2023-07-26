# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is the entry point for launching experiments with Implicitron.

Launch Training
---------------
Experiment config .yaml files are located in the
`configs/` folder. To launch an experiment,
specify the name of the file. Specific config values can also be overridden
from the command line, for example:

```
python experiment.py --config-name base_config.yaml override.param.one=42 override.param.two=84
```

Main functions
---------------
- The Experiment class defines `run` which creates the model, optimizer, and other
  objects used in training, then starts TrainingLoop's `run` function.
- TrainingLoop takes care of the actual training logic: forward and backward passes,
  evaluation and testing, as well as model checkpointing, visualization, and metric
  printing.

Outputs
--------
The outputs of the experiment are saved and logged in multiple ways:
  - Checkpoints:
        Model, optimizer and stats are stored in the directory
        named by the `exp_dir` key from the config file / CLI parameters.
  - Stats
        Stats are logged and plotted to the file "train_stats.pdf" in the
        same directory. The stats are also saved as part of the checkpoint file.
  - Visualizations
        Predictions are plotted to a visdom server running at the
        port specified by the `visdom_server` and `visdom_port` keys in the
        config file.
"""


import hydra
import logging
import os
import socket
import torch
import warnings

from accelerate import Accelerator
from dataclasses import field
from omegaconf import DictConfig, OmegaConf
from packaging import version

# pytorch3d imports
from pytorch3d.implicitron.dataset.data_source import DataSourceBase
from pytorch3d.implicitron.dataset.data_loader_map_provider import BatchConditioningType
from pytorch3d.implicitron.tools.config import (
    Configurable,
    expand_args_fields,
    remove_unused_components,
    run_auto_creation,
)

# trainer imports
from trainer.model_factory import ModelFactoryBase
from trainer.optimizer_factory import OptimizerFactoryBase
from trainer.training_loop import TrainingLoopBase
from trainer.utils import seed_all_random_engines

# (HoloDiffusion) populate the registry with our own classes
from holo_diffusion.holo_diffusion_model import HoloDiffusionModel


def setup_logging():
    hostname_ = socket.gethostname()
    logging.basicConfig(
        format="[%(levelname)s] {} %(asctime)s: %(message)s".format(hostname_),
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


setup_logging()

logger = logging.getLogger(__name__)

# workaround for https://github.com/facebookresearch/hydra/issues/2262
_RUN = hydra.types.RunMode.RUN

if version.parse(hydra.__version__) < version.Version("1.1"):
    raise ValueError(
        f"Hydra version {hydra.__version__} is too old."
        " (Implicitron requires version 1.1 or later.)"
    )

try:
    # only makes sense in FAIR cluster
    import pytorch3d.implicitron.fair_cluster.slurm  # noqa: F401
except ModuleNotFoundError:
    pass

no_accelerate = os.environ.get("PYTORCH3D_NO_ACCELERATE") is not None


class Experiment(Configurable):  # pyre-ignore: 13
    """
    This class is at the top level of Implicitron's config hierarchy. Its
    members are high-level components necessary for training an implicitron model.

    Members:
        data_source: An object that produces datasets and dataloaders.
        model_factory: An object that produces an implicit rendering model
        optimizer_factory: An object that produces the optimizer and lr
            scheduler.
        training_loop: An object that runs training given the outputs produced
            by the data_source, model_factory and optimizer_factory.
        seed: A random seed to ensure reproducibility.
        detect_anomaly: Whether torch.autograd should detect anomalies. Useful
            for debugging, but might slow down the training.
        exp_dir: Root experimentation directory. Checkpoints and training stats
            will be saved here.
    """

    data_source: DataSourceBase
    data_source_class_type: str = "ImplicitronDataSource"
    model_factory: ModelFactoryBase
    model_factory_class_type: str = "ImplicitronModelFactory"
    optimizer_factory: OptimizerFactoryBase
    optimizer_factory_class_type: str = "ImplicitronOptimizerFactory"
    training_loop: TrainingLoopBase
    training_loop_class_type: str = "ImplicitronTrainingLoop"

    disable_testing: bool = False
    disable_validation: bool = False

    seed: int = 42
    detect_anomaly: bool = False
    exp_dir: str = "/home/karnewar/projects/holo_diffusion/release_experiments"

    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},  # Make hydra not change the working dir.
            "output_subdir": None,  # disable storing the .hydra logs
            "mode": _RUN,
        }
    )

    def __post_init__(self):
        seed_all_random_engines(
            self.seed
        )  # Set all random engine seeds for reproducibility

        run_auto_creation(self)

    def run(self) -> None:
        # Initialize the accelerator if desired.
        if no_accelerate:
            seed = self.seed
            accelerator = None
            device = torch.device("cuda:0")
        else:
            accelerator = Accelerator(device_placement=False)
            logger.info(accelerator.state)
            seed = self.seed + 10000 * accelerator.process_index
            if str(accelerator.state.distributed_type) == "DistributedType.NO":
                # running locally
                device = "cuda:0"
            else:
                device = accelerator.device
            torch.cuda.set_device(device)

        logger.info(f"Seed = {seed}")
        seed_all_random_engines(seed)

        logger.info(f"Running experiment on device: {device}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # set the debug mode
        if self.detect_anomaly:
            logger.info("Anomaly detection!")
        torch.autograd.set_detect_anomaly(self.detect_anomaly)

        # Initialize the datasets and dataloaders.
        for conditioning_type in (
            "train_conditioning_type",
            "val_conditioning_type",
            "test_conditioning_type",
        ):
            conditioning_type_value = getattr(
                self.data_source.data_loader_map_provider, conditioning_type
            )
            if isinstance(conditioning_type_value, str):
                setattr(
                    self.data_source.data_loader_map_provider,
                    conditioning_type,
                    BatchConditioningType(conditioning_type_value.lower()),
                )
        datasets, dataloaders = self.data_source.get_datasets_and_dataloaders()

        # Init the model and the corresponding Stats object.
        model = self.model_factory(
            accelerator=accelerator,
            exp_dir=self.exp_dir,
        )
        stats = self.training_loop.load_stats(
            log_vars=model.log_vars,
            exp_dir=self.exp_dir,
            resume=self.model_factory.resume,
            resume_epoch=self.model_factory.resume_epoch,  # pyre-ignore [16]
        )
        start_epoch = stats.epoch + 1
        model.to(device)

        # Init the optimizer and LR scheduler.
        optimizer, scheduler = self.optimizer_factory(
            accelerator=accelerator,
            exp_dir=self.exp_dir,
            last_epoch=start_epoch,
            model=model,
            resume=self.model_factory.resume,
            resume_epoch=self.model_factory.resume_epoch,
        )

        # Wrap all modules in the distributed library
        # Note: we don't pass the scheduler to prepare as it
        # doesn't need to be stepped at each optimizer step
        train_loader = dataloaders.train
        val_loader = dataloaders.val
        test_loader = dataloaders.test

        if accelerator is not None:
            datum = next(iter(train_loader))
            image_rgb = datum.image_rgb.to(device)
            camera = datum.camera.to(device)
            fg_probability = datum.fg_probability.to(device)
            mask_crop = datum.mask_crop.to(device)
            with torch.no_grad():
                model(
                    image_rgb=image_rgb,
                    camera=camera,
                    sequence_name=datum.sequence_name,
                    fg_probability=fg_probability,
                    mask_crop=mask_crop,
                )

            # HACK: we add missing attributes to the data_loaders for accelerator
            train_loader.batch_sampler.sampler = train_loader.batch_sampler
            train_loader.batch_sampler.drop_last = False
            val_loader.batch_sampler.sampler = val_loader.batch_sampler
            val_loader.batch_sampler.drop_last = False

            (
                model,
                optimizer,
                train_loader,
                val_loader,
            ) = accelerator.prepare(model, optimizer, train_loader, val_loader)

            del datum, image_rgb, camera, fg_probability, mask_crop

        # pyre-fixme[16]: Optional type has no attribute `is_multisequence`.
        if not self.training_loop.evaluator.is_multisequence:
            all_train_cameras = self.data_source.all_train_cameras
        else:
            all_train_cameras = None

        # Enter the main training loop.
        val_loader = None if self.disable_validation else val_loader
        test_loader = None if self.disable_testing else test_loader
        self.training_loop.run(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            # pyre-ignore[6]
            train_dataset=datasets.train,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            all_train_cameras=all_train_cameras,
            accelerator=accelerator,
            device=device,
            exp_dir=self.exp_dir,
            stats=stats,
            seed=seed,
        )


def _setup_envvars_for_cluster() -> bool:
    """
    Prepares to run on cluster if relevant.
    Returns whether FAIR cluster in use.
    """
    try:
        import submitit
    except ImportError:
        return False
    return True


def dump_cfg(exp_dir: str, cfg: DictConfig) -> None:
    remove_unused_components(cfg)
    # dump the exp config to the exp dir
    os.makedirs(exp_dir, exist_ok=True)
    try:
        cfg_filename = os.path.join(exp_dir, "expconfig.yaml")
        OmegaConf.save(config=cfg, f=cfg_filename)
    except PermissionError:
        warnings.warn("Can't dump config due to insufficient permissions!")


expand_args_fields(Experiment)
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config", node=Experiment)


@hydra.main(config_path="./configs/", config_name="default_config")
def experiment(cfg: DictConfig) -> None:
    # CUDA_VISIBLE_DEVICES must have been set.
    print("Experiment!")

    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if not _setup_envvars_for_cluster():
        logger.info("Running locally")

    # create a new experiment
    experiment = Experiment(**cfg)
    # dume the config to the exp dir
    dump_cfg(experiment.exp_dir, cfg)
    # run the experiment
    experiment.run()


if __name__ == "__main__":
    experiment()
