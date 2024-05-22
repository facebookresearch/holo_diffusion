# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import Any, Iterator, List, Optional

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, ProfilerActivity
from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.evaluation.evaluator import EvaluatorBase
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.models.generic_model import EvaluationMode
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.implicitron.tools.stats import Stats


from .timer import Timer
from .utils import (
    get_optimizer_discriminator_path,
    seed_all_random_engines,
    use_seed,
    path_mkdir,
)
from .optimizer_factory import OptimizerFactoryBase

logger = logging.getLogger(__name__)

PROFILE = False


# pyre-fixme[13]: Attribute `evaluator` is never initialized.
class TrainingLoopBase(ReplaceableBase):
    """
    Members:
        evaluator: An EvaluatorBase instance, used to evaluate training results.
    """

    evaluator: Optional[EvaluatorBase]
    evaluator_class_type: Optional[str] = "ImplicitronEvaluator"

    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        train_dataset: Dataset,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        **kwargs,
    ) -> None:
        raise NotImplementedError()

    def load_stats(
        self,
        log_vars: List[str],
        exp_dir: str,
        resume: bool = True,
        resume_epoch: int = -1,
        **kwargs,
    ) -> Stats:
        raise NotImplementedError()


@registry.register
class ImplicitronTrainingLoop(TrainingLoopBase):
    """
    Members:
        eval_only: If True, only run evaluation using the test dataloader.
        max_epochs: Train for this many epochs. Note that if the model was
            loaded from a checkpoint, we will restart training at the appropriate
            epoch and run for (max_epochs - checkpoint_epoch) epochs.
        store_checkpoints: If True, store model and optimizer state checkpoints.
        store_checkpoints_purge: If >= 0, remove any checkpoints older than
            this many epochs (inclusive, so =1 means only the current epoch is kept)
        test_interval: Evaluate on a test dataloader each `test_interval` epochs.
        test_when_finished: If True, evaluate on a test dataloader when training
            completes.
        validation_interval: Validate each `validation_interval` epochs.
        clip_grad: Optionally clip the gradient norms.
            If set to a value <=0.0, no clipping
        metric_print_interval: The batch interval at which the stats should be
            logged.
        visualize_interval: The batch interval at which the visualizations
            should be plotted
        visdom_env: The name of the Visdom environment to use for plotting.
        visdom_port: The Visdom port.
        visdom_server: Address of the Visdom server.
    """

    # Parameters of the outer training loop.
    eval_only: bool = False
    max_epochs: int = 1000
    store_checkpoints: bool = True
    store_stats: bool = True
    store_checkpoints_purge: int = 1
    test_interval: int = -1
    test_when_finished: bool = False
    validation_interval: int = 1

    # Gradient clipping.
    clip_grad: float = 0.0

    # Visualization/logging parameters.
    metric_print_interval: int = 5
    visualize_interval: int = 1000
    plot_stats_to_visdom: bool = True
    visdom_env: str = ""
    visdom_port: int = int(os.environ.get("VISDOM_PORT", 8097))
    visdom_server: str = "http://127.0.0.1"

    whole_dataset_batch: bool = False
    n_batches_in_epoch: int = 1
    deterministic_validation: bool = False
    log_val_metrics: bool = True
    save_val_img_predictions: bool = False
    n_save_per_seq: int = 4

    profile: bool = False
    trace_file_fw: str = "trace_fw.json"
    trace_file_bw: str = "trace_bw.json"

    # Experimental parameters - deprecated
    checkpoint_tmp_mode: bool = True

    def __post_init__(self):
        run_auto_creation(self)
        self._epoch2subscribers = defaultdict(lambda: [])
        if self.plot_stats_to_visdom:
            print(f"plotting to visdom: env={self.visdom_env}, port={self.visdom_port}")
        if self.save_val_img_predictions:
            self.image_loggers = {}

    def run(
        self,
        *,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        train_dataset: Dataset,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        optimizer_factory: OptimizerFactoryBase,
        optimizer_discriminator: Optional[torch.optim.Optimizer],
        scheduler: Any,
        accelerator: Optional[Accelerator],
        device: torch.device,
        exp_dir: str,
        stats: Stats,
        seed: int,
        **kwargs,
    ):
        """
        Entry point to run the training and validation loops
        based on the specified config file.
        """
        start_epoch = stats.epoch + 1
        assert scheduler.last_epoch == stats.epoch + 1
        assert scheduler.last_epoch == start_epoch

        # only run evaluation on the test dataloader
        if self.eval_only:
            if test_loader is not None:
                if self.evaluator is None:
                    raise ValueError("Cannot evaluate if evaluator is not set.")
                self.evaluator.run(
                    dataloader=test_loader,
                    device=device,
                    dump_to_json=True,
                    epoch=stats.epoch,
                    exp_dir=exp_dir,
                    model=model,
                )
                return
            else:
                raise ValueError(
                    "Cannot evaluate and dump results to json, no test data provided."
                )

        # collect all subscribers for certain optimisation epochs
        model.apply(self._collect_epoch_subscribers)

        # get the whole dataset in one batch and construct a train_loader from it
        if self.whole_dataset_batch:
            # pyre-ignore[6, 58, 29]
            if 0 < model.n_train_target_views < len(train_dataset):
                raise ValueError(
                    #  # pyre-ignore[6]
                    "If you want to train on the whole dataset in one batch, "
                    "`model.n_train_target_views` must be set to include "
                    f"the whole batch of {len(train_dataset)} views. "
                    "E.g. set it to 0."
                )
            # pyre-ignore[9]
            train_loader = _WholeDatasetLoader(
                train_dataset, self.n_batches_in_epoch, device
            )

        # loop through epochs
        for epoch in range(start_epoch, self.max_epochs):
            # publish current epoch and if some model parameters changed
            # create a new optimizer and scheduler
            change = self._publish_epoch(epoch)
            if change:
                optimizer, scheduler = optimizer_factory(
                    accelerator=accelerator,
                    last_epoch=epoch,
                    model=model,
                    resume=False,
                    resume_epoch=True,
                )
                if accelerator:
                    optimizer = accelerator.prepare(optimizer)

            # automatic new_epoch and plotting of stats at every epoch start
            with stats:
                # Make sure to re-seed random generators to ensure reproducibility
                # even after restart.
                seed_all_random_engines(seed + epoch)

                cur_lr = float(scheduler.get_last_lr()[-1])
                logger.debug(f"scheduler lr = {cur_lr:1.2e}")

                # train loop
                self._training_or_validation_epoch(
                    accelerator=accelerator,
                    device=device,
                    epoch=epoch,
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    optimizer_discriminator=optimizer_discriminator,
                    stats=stats,
                    validation=False,
                )

                # val loop (optional)
                if val_loader is not None and epoch % self.validation_interval == 0:
                    self._training_or_validation_epoch(
                        accelerator=accelerator,
                        device=device,
                        epoch=epoch,
                        loader=val_loader,
                        model=model,
                        optimizer=optimizer,
                        optimizer_discriminator=optimizer_discriminator,
                        stats=stats,
                        validation=True,
                        seed=seed if self.deterministic_validation else None,
                    )

                # eval loop (optional)
                if (
                    test_loader is not None
                    and self.test_interval > 0
                    and epoch % self.test_interval == 0
                ):
                    if self.evaluator is None:
                        raise ValueError("Cannot evaluate if evaluator is not set.")
                    self.evaluator.run(
                        device=device,
                        dataloader=test_loader,
                        model=model,
                    )

                assert stats.epoch == epoch, "inconsistent stats!"
                self._checkpoint(
                    accelerator,
                    epoch,
                    exp_dir,
                    model,
                    optimizer,
                    stats,
                    optimizer_discriminator=optimizer_discriminator,
                )

                scheduler.step()
                new_lr = float(scheduler.get_last_lr()[-1])
                if new_lr != cur_lr:
                    logger.info(f"LR change! {cur_lr} -> {new_lr}")

        if self.test_when_finished:
            if test_loader is not None:
                if self.evaluator is None:
                    raise ValueError("Cannot evaluate if evaluator is not set.")
                self.evaluator.run(
                    device=device,
                    dump_to_json=True,
                    epoch=stats.epoch,
                    exp_dir=exp_dir,
                    dataloader=test_loader,
                    model=model,
                )
            else:
                raise ValueError(
                    "Cannot evaluate and dump results to json, no test data provided."
                )

        if self.save_val_img_predictions:
            [im_logger.save_video() for im_logger in self.image_loggers.values()]

    def load_stats(
        self,
        log_vars: List[str],
        exp_dir: str,
        resume: bool = True,
        resume_epoch: int = -1,
        **kwargs,
    ) -> Stats:
        """
        Load Stats that correspond to the model's log_vars and resume_epoch.

        Args:
            log_vars: A list of variable names to log. Should be a subset of the
                `preds` returned by the forward function of the corresponding
                ImplicitronModelBase instance.
            exp_dir: Root experiment directory.
            resume: If False, do not load stats from the checkpoint speci-
                fied by resume and resume_epoch; instead, create a fresh stats object.

        stats: The stats structure (optionally loaded from checkpoint)
        """
        # Init the stats struct
        visdom_env_charts = (
            vis_utils.get_visdom_env(self.visdom_env, exp_dir) + "_charts"
        )
        stats = Stats(
            # log_vars should be a list, but OmegaConf might load them as ListConfig
            list(log_vars),
            plot_file=os.path.join(exp_dir, "train_stats.pdf"),
            visdom_env=visdom_env_charts,
            visdom_server=self.visdom_server,
            visdom_port=self.visdom_port,
            do_plot=self.plot_stats_to_visdom,
        )

        model_path = None
        if resume:
            if resume_epoch > 0:
                model_path = model_io.get_checkpoint(exp_dir, resume_epoch)
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(
                        f"Cannot find stats from epoch {resume_epoch}."
                    )
            else:
                model_path = model_io.find_last_checkpoint(exp_dir)

        if model_path is not None:
            stats_path = model_io.get_stats_path(model_path)
            stats_load = model_io.load_stats(stats_path)

            # Determine if stats should be reset
            if resume:
                if stats_load is None:
                    logger.warning("\n\n\n\nCORRUPT STATS -> clearing stats\n\n\n\n")
                    last_epoch = model_io.parse_epoch_from_model_path(model_path)
                    logger.info(f"Estimated resume epoch = {last_epoch}")

                    # Reset the stats struct
                    for _ in range(last_epoch + 1):
                        stats.new_epoch()
                    assert last_epoch == stats.epoch
                else:
                    logger.info(f"Found previous stats in {stats_path} -> resuming.")
                    stats = stats_load

                # Update stats properties incase it was reset on load
                stats.visdom_env = visdom_env_charts
                stats.visdom_server = self.visdom_server
                stats.visdom_port = self.visdom_port
                stats.plot_file = os.path.join(exp_dir, "train_stats.pdf")
                stats.do_plot = self.plot_stats_to_visdom
                stats.synchronize_logged_vars(log_vars)
            else:
                logger.info("Clearing stats")

        return stats

    @use_seed()
    def _training_or_validation_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        stats: Stats,
        validation: bool,
        *,
        accelerator: Optional[Accelerator],
        optimizer_discriminator: Optional[torch.optim.Optimizer] = None,
        bp_var: str = "objective",
        device: torch.device,
        **kwargs,
    ) -> None:
        """
        This is the main loop for training and evaluation including:
        model forward pass, loss computation, backward pass and visualization.

        Args:
            epoch: The index of the current epoch
            loader: The dataloader to use for the loop
            model: The model module optionally loaded from checkpoint
            optimizer: The optimizer module optionally loaded from checkpoint
            stats: The stats struct, also optionally loaded from checkpoint
            validation: If true, run the loop with the model in eval mode
                and skip the backward pass
            accelerator: An optional Accelerator instance.
            bp_var: The name of the key in the model output `preds` dict which
                should be used as the loss for the backward pass.
            device: The device on which to run the model.
        """

        if validation:
            model.eval()
            trainmode = "val"
        else:
            model.train()
            trainmode = "train"

        # get the visdom env name
        visdom_env_imgs = stats.visdom_env + "_images_" + trainmode
        viz = vis_utils.get_visdom_connection(
            server=stats.visdom_server, port=stats.visdom_port
        )

        # Iterate through the batches
        n_batches = len(loader)
        loader_iter = iter(loader)

        t_start = time.time()
        for it in range(n_batches):
            with Timer(name="loader_iter", logger=None) as load_timer:
                net_input = next(loader_iter)
            load_time = (
                load_timer.timers["loader_iter"]
                / load_timer.timers_n_ticks["loader_iter"]
            )

            last_iter = it == n_batches - 1

            # move to gpu where possible (in place)
            net_input = net_input.to(device)

            # run the forward pass
            if not validation:
                optimizer.zero_grad()

                if self.profile:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        with_stack=True,
                    ) as prof:
                        preds = model(
                            **{**net_input, "evaluation_mode": EvaluationMode.TRAINING}
                        )
                    print(f"Writing trace file {self.trace_file_fw}.")
                    prof.export_chrome_trace(self.trace_file_fw)
                else:
                    preds = model(
                        **{**net_input, "evaluation_mode": EvaluationMode.TRAINING}
                    )
            else:
                with torch.no_grad():
                    preds = model(
                        **{**net_input, "evaluation_mode": EvaluationMode.EVALUATION}
                    )

            # make sure we dont overwrite something
            assert all(k not in preds for k in net_input.keys())
            # merge everything into one big dict
            preds.update(net_input)
            preds["sec/load_it"] = load_time

            if trainmode == "train" or self.log_val_metrics:
                # update the stats logger
                stats.update(preds, time_start=t_start, stat_set=trainmode)
                # pyre-ignore [16]
                assert stats.it[trainmode] == it, "inconsistent stat iteration number!"

                # print textual status update
                if it % self.metric_print_interval == 0 or last_iter:
                    std_out = stats.get_status_string(n_batches, trainmode)
                    logger.info(std_out)

            # visualize results
            if (
                (accelerator is None or accelerator.is_local_main_process)
                and self.visualize_interval > 0
                and it % self.visualize_interval == 0
            ):
                prefix = f"e{stats.epoch}_it{it}"
                if isinstance(
                    model, torch.nn.parallel.distributed.DistributedDataParallel
                ):
                    viz_model = model.module
                else:
                    viz_model = model
                if hasattr(viz_model, "visualize"):
                    # pyre-ignore [29]
                    viz_model.visualize(viz, visdom_env_imgs, preds, prefix)

            if not validation:
                # optimizer step
                loss = preds[bp_var]
                assert torch.isfinite(loss).all(), "Non-finite loss!"
                if loss.requires_grad:
                    with Timer("Backward_pass", disable=not PROFILE, cuda_sync=True):
                        # backprop
                        if self.profile:
                            with profile(
                                activities=[
                                    ProfilerActivity.CPU,
                                    ProfilerActivity.CUDA,
                                ],
                                with_stack=True,
                            ) as prof:
                                if accelerator is None:
                                    loss.backward()
                                else:
                                    accelerator.backward(loss)
                            print(f"Writing trace file {self.trace_file_bw}.")
                            prof.export_chrome_trace(self.trace_file_bw)
                        else:
                            if accelerator is None:
                                loss.backward()
                            else:
                                accelerator.backward(loss)
                        if self.clip_grad > 0.0:
                            # Optionally clip the gradient norms.
                            total_norm = torch.nn.utils.clip_grad_norm(
                                model.parameters(), self.clip_grad
                            )
                            if total_norm > self.clip_grad:
                                logger.debug(
                                    f"Clipping gradient: {total_norm}"
                                    + f" with coef {self.clip_grad / float(total_norm)}."
                                )

                    with Timer("Optimizer_step", disable=not PROFILE, cuda_sync=True):
                        optimizer.step()

                else:
                    warnings.warn(
                        "The loss does not require grad, optimization will be skipped."
                    )

                if optimizer_discriminator is not None:
                    assert (
                        "loss_gan_dis" in preds
                    ), "loss_gan_dis not in preds! In spite of using a discriminator"
                    optimizer_discriminator.zero_grad()
                    preds["loss_gan_dis"].backward()
                    optimizer_discriminator.step()

            elif (
                self.save_val_img_predictions
                and it == 0
                and (accelerator is None or accelerator.is_local_main_process)
            ):
                self._save_images(model, preds, epoch, stats)

    def _save_images(self, model, preds, epoch, stats):
        # Logging visuals
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            viz_model = model.module
        else:
            viz_model = model

        if hasattr(viz_model, "log_img_names"):
            N, is_final = self.n_save_per_seq, epoch == self.max_epochs - 1
            if len(self.image_loggers) == 0:
                # We create the loggers
                exp_dir = stats.plot_file.rsplit("/", 1)[0]
                log_dir = path_mkdir(f"{exp_dir}/visual_logs")
                for k in viz_model.log_img_names.keys():
                    try:
                        im = preds[k].view(
                            -1, viz_model.n_train_target_views, *preds[k].shape[1:]
                        )
                    except RuntimeError:  # reshape fails
                        logger.warning(f"Could not reshape logged image {k}.")
                        im = preds[k].view(1, -1, *preds[k].shape[1:])
                    # self.image_loggers[k] = ImageLogger(
                    #     log_dir / k, im[:, :N].reshape(-1, *preds[k].shape[1:])
                    # )
                    pass

            for k, v in viz_model.log_img_names.items():
                try:
                    renders = preds[v].view(
                        -1, viz_model.n_train_target_views, *preds[v].shape[1:]
                    )
                except RuntimeError:  # reshape fails
                    logger.warning(f"Could not reshape logged image {v}.")
                    renders = preds[v].view(1, -1, *preds[v].shape[1:])
                self.image_loggers[k].save(
                    renders[:, :N].reshape(-1, *preds[v].shape[1:]),
                    it=None if is_final else epoch,
                )

    def _checkpoint(
        self,
        accelerator: Optional[Accelerator],
        epoch: int,
        exp_dir: str,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        stats: Stats,
        optimizer_discriminator: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Save a model and its corresponding Stats object to a file, if
        `self.store_checkpoints` is True. In addition, if
        `self.store_checkpoints_purge` is True, remove any checkpoints older
        than `self.store_checkpoints_purge` epochs old (inclusive, so =1 means only the
        current epoch is kept).
        """

        if accelerator is None or (
            accelerator.is_local_main_process and accelerator.is_main_process
        ):
            if self.store_checkpoints:
                outfile = model_io.get_checkpoint(exp_dir, epoch)
                unwrapped_model = (
                    model if accelerator is None else accelerator.unwrap_model(model)
                )
                try:
                    model_io.safe_save_model(
                        unwrapped_model, stats, outfile, optimizer=optimizer
                    )
                    # also store the discriminator
                    if optimizer_discriminator is not None:
                        opt_disc_path = get_optimizer_discriminator_path(outfile)
                        logger.info(f"Saving discriminator optim to {opt_disc_path}")
                        torch.save(optimizer_discriminator.state_dict(), opt_disc_path)

                except Exception as msg:
                    logger.warn(
                        "Could not save model, make sure there is enough space!"
                        f" Error stacktrace: {msg}"
                    )

                else:
                    if self.store_checkpoints_purge > 0:
                        for old_k in range(epoch - self.store_checkpoints_purge + 1):
                            model_io.purge_epoch(exp_dir, old_k)
                            if optimizer_discriminator is not None:
                                # also purge the discriminator if any
                                opt_disc_path = get_optimizer_discriminator_path(
                                    model_io.get_checkpoint(exp_dir, old_k)
                                )
                                if os.path.isfile(opt_disc_path):
                                    logger.info("deleting %s" % opt_disc_path)
                                    os.remove(opt_disc_path)

            elif self.store_stats:
                outfile = model_io.get_checkpoint(exp_dir, epoch)
                model_io.save_stats(stats, outfile)

    def _collect_epoch_subscribers(self, module: torch.nn.Module) -> None:
        """
        Collects all modules that wish to subscribe to epoch updates.

        Module is considered to wish to subscribe if it has subscribe_to_epochs
        method, which should return list of epochs that it wishes to subscribe to
        and a function which will run on that epoch.

        Args:
            module: a module that will be checked for its subscription wishes
        Returns:
            nothing
        """
        subscribe_to_epochs = getattr(module, "subscribe_to_epochs", None)
        if callable(subscribe_to_epochs):
            wanted_epochs, apply_func = subscribe_to_epochs()
            for epoch in wanted_epochs:
                # pyre-ignore[16]
                self._epoch2subscribers[epoch].append(apply_func)

    def _publish_epoch(self, epoch: int) -> bool:
        """
        Method which publishes current epoch for subscribed modules.

        Args:
            epoch: current epoch
        Returns:
            True if change has happened else False
        """
        # pyre-ignore [16]
        change = False
        for subscriber in self._epoch2subscribers[epoch]:
            change = change or subscriber(epoch)
        return change


HoloXTrainingLoop = ImplicitronTrainingLoop


class _WholeDatasetLoader:
    """
    Loades the whole dataset on device and provides and iterator over it.
    Returns `n_batches_in_epoch` batches, where one batch is the whole
    dataset.

    Members:
        train_dataset: dataset to load
        n_batches_in_epoch: how many batches to have in an epoch.
        device: torch.device on which to load the dataset,
    """

    def __init__(
        self, train_dataset: Dataset, n_batches_in_epoch: int, device: torch.device
    ) -> None:
        self.n_batches_in_epoch = n_batches_in_epoch
        # pyre-ignore[6]
        train_data = [train_dataset[i] for i in range(len(train_dataset))]
        self.train_dataset_batch = train_data[0].collate(train_data).to(device)

    def __iter__(self) -> Iterator[FrameData]:
        return itertools.repeat(self.train_dataset_batch, self.n_batches_in_epoch)

    def __len__(self) -> int:
        return self.n_batches_in_epoch
