# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.dataset.data_source import DataSourceBase
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.tools.config import Configurable
from typing import Optional, Tuple, Type


def _get_config_from_experiment_directory(experiment_directory: str) -> DictConfig:
    cfg_file = os.path.join(experiment_directory, "expconfig.yaml")
    config = OmegaConf.load(cfg_file)
    return config


def load_experiment(
    ExperimentClass: Type[Configurable],
    exp_dir: str,
    restrict_sequence_name: Optional[str] = None,
    render_size: Optional[Tuple[int, int]] = None,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Configurable, ImplicitronModelBase, DataSourceBase]:
    # make the schema
    schema = OmegaConf.structured(ExperimentClass)

    # Get the config from the experiment_directory,
    # and overwrite relevant fields
    loaded_config = _get_config_from_experiment_directory(exp_dir)
    config = OmegaConf.merge(schema, loaded_config)
    config.exp_dir = exp_dir

    # important so that the CO3D dataset gets loaded in full
    data_source_args = config.data_source_ImplicitronDataSource_args
    if "dataset_map_provider_JsonIndexDatasetMapProvider_args" in data_source_args:
        dataset_args = (
            data_source_args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        )
        dataset_args.test_on_train = False
        if restrict_sequence_name is not None:
            dataset_args.restrict_sequence_name = restrict_sequence_name
    if "dataset_map_provider_JsonIndexDatasetMapProviderV2_args" in data_source_args:
        dataset_args = (
            data_source_args.dataset_map_provider_JsonIndexDatasetMapProviderV2_args
        )
        dataset_args.test_on_train = False
        if restrict_sequence_name is not None:
            dataset_args.restrict_sequence_name = restrict_sequence_name
        dataset_args.dataset_JsonIndexDataset_args.limit_sequences_to = -1

    # Set the rendering image size
    model_factory_args = config.model_factory_ImplicitronModelFactory_args
    model_factory_args.force_resume = True
    model_args = model_factory_args.model_HoloDiffusionModel_args
    if render_size is not None:
        model_args.render_image_width = render_size[0]
        model_args.render_image_height = render_size[1]

    # Load the previously trained model
    config.seed = seed
    experiment = ExperimentClass(**config)
    model = experiment.model_factory(exp_dir=exp_dir)
    model.to(device)

    # Setup the dataset
    data_source = experiment.data_source

    # return the three things
    return experiment, model, data_source
