from datetime import datetime
import threading
from typing import Dict
import os
import easybar
import shutil
import pickle

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.dl import DataParallelEngine, DistributedDataParallelEngine
from catalyst.data.loader import ILoaderWrapper

import ipdb
import nibabel as nib
import numpy as np

import torch
from torch.optim.lr_scheduler import (
    MultiStepLR,
    OneCycleLR,
    CosineAnnealingLR,
    ChainedScheduler,
    CyclicLR,
    StepLR,
    ConstantLR,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dice import faster_dice, DiceLoss
from meshnet import enMesh_checkpoint, enMesh
from mongoslabs.gencoords import CoordsGenerator
import monai.networks.nets as nets

from mindfultensors.redisloader import (
    create_client,
    collate_subcubes,
    mcollate,
    DBBatchSampler,
    RedisDataset,
    mtransform,
    qnormalize,
    unit_interval_normalize,
)

# SEED = 0
# utils.set_global_seed(SEED)
# utils.prepare_cudnn(deterministic=False, benchmark=False) # crashes everything

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
#os.environ["NCCL_P2P_LEVEL"] = "NVL"

volume_shape = [256] * 3
MAXSHAPE = 300


n_classes = 32

REDISHOST = "10.245.12.57"  # "arctrdcn017.rs.gsu.edu"
DBKEY = "db1"
INDEX_ID = "id"
config_file = "modelAE.json"
#WANDBTEAM = "neuroneural"


# CustomRunner – PyTorch for-loop decomposition
# https://github.com/catalyst-team/catalyst#minimal-examples
class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir: str,
        wandb_project: str,
        wandb_experiment: str,
        model_path: str,
        n_channels: int,
        n_classes: int,
        optimize_inline: bool,
        validation_percent: float,
        onecycle_lr: float,
        rmsprop_lr: float,
        num_subcubes: int,
        num_volumes: int,
        epoch_length: int,
        n_epochs: int,
        client_creator,
        off_brain_weight: float,
        prefetches=8,
        db_name=DBKEY,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
    ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.optimize_inline = optimize_inline
        self.onecycle_lr = onecycle_lr
        self.rmsprop_lr = rmsprop_lr
        self.prefetches = prefetches
        self.db_name = db_name
        self.shape = subvolume_shape[0]
        self.num_subcubes = num_subcubes
        self.num_volumes = num_volumes
        self.epoch_length = epoch_length
        self.n_epochs = n_epochs
        self.off_brain_weight = off_brain_weight
        self.client_creator = client_creator
        self.funcs = None
        self.collate = None

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            return dl.DistributedDataParallelEngine(
                # mixed_precision='fp16',
                # ddp_kwargs={"find_unused_parameters": True, "backend": "nccl"}
                process_group_kwargs={"backend": "nccl"}
            )
        else:
            return dl.GPUEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            # "tensorboard": dl.TensorboardLogger(logdir=self._logdir,
            #                                     log_batch_metrics=True),
            "wandb": dl.WandbLogger(
                project=self.wandb_project,
                name=self.wandb_experiment,
                #entity=WANDBTEAM,
                log_batch_metrics=True,
            ),
        }

    @property
    def stages(self):
        return ["train"]

    @property
    def num_epochs(self) -> int:
        return self.n_epochs

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        random_data = os.urandom(4)
        SEED = int.from_bytes(random_data, byteorder="big")
        utils.set_global_seed(SEED)
        return SEED

    def get_stage_len(self) -> int:
        return 1

    def get_loaders(self):
        self.funcs = {
            "createclient": self.client_creator.create_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }

        self.collate = (
            self.funcs["mycollate_full"]
            if self.shape == 256
            else self.funcs["mycollate"]
        )

        tdataset = RedisDataset(
            range(self.epoch_length),
            self.funcs["mytransform"],
            self.db_name,
            normalize=qnormalize,
        )

        tsampler = (
            DistributedSamplerWrapper(
                DBBatchSampler(tdataset, batch_size=self.num_volumes)
            )
            if self.engine.is_ddp
            else DBBatchSampler(tdataset, batch_size=self.num_volumes)
        )

        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                sampler=tsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                prefetch_factor=3,
                num_workers=self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        return {"train": tdataloader}

    '''
    def get_model(self):
        if self.shape > MAXSHAPE:
            model = enMesh(
                in_channels=1,
                n_classes=self.n_classes,
                channels=self.n_channels,
                config_file=config_file,
                optimize_inline=self.optimize_inline,
            )
        else:
            model = enMesh_checkpoint(
                in_channels=1,
                n_classes=self.n_classes,
                channels=self.n_channels,
                config_file=config_file,
            )
        return model
    '''
    def get_model(self):
        model = nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        return model

    def get_criterion(self):
        # class_weight = torch.FloatTensor(
        #     [self.off_brain_weight] + [1.0] * (self.n_classes - 1)
        # ).to(self.engine.device)
        # criterion = torch.nn.CrossEntropyLoss(
        #     weight=class_weight, label_smoothing=0.01
        # )
        criterion = DiceLoss()
        return criterion

    def get_optimizer(self, model):
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.rmsprop_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=100,
            pct_start=0.2,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )
        return scheduler

    def get_callbacks(self):
        checkpoint_params = {"save_best": True, "metric_key": "macro_dice"}
        if self.model_path:
            # checkpoint_params = {"resume_model": self.model_path}
            checkpoint_params.update({"resume_model": self.model_path})
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "macro_dice", "learning rate"]
        }

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        for key in ["loss", "macro_dice", "learning rate"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):
        # unpack the batch
        sample, label = batch

        # run model forward/backward pass
        if self.model.training:
            if self.shape > MAXSHAPE:
                if self.engine.is_ddp:
                    with self.model.no_sync():
                        loss, y_hat = self.model.forward(
                            x=sample,
                            y=label,
                            loss=self.criterion,
                            verbose=False,
                        )
                    torch.distributed.barrier()
                else:
                    loss, y_hat = self.model.forward(
                        x=sample, y=label, loss=self.criterion, verbose=False
                    )
            else:
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
                loss.backward()
            if not self.optimize_inline:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
        with torch.inference_mode():
            result = torch.squeeze(torch.argmax(y_hat, 1)).long()
            labels = torch.squeeze(label)

            dice = torch.mean(
                faster_dice(result, labels, range(self.n_classes))
            )

        self.batch_metrics.update(
            {
                "loss": loss,
                "macro_dice": dice,
                "learning rate": torch.tensor(
                    self.optimizer.param_groups[0]["lr"]
                ),
            }
        )

        for key in ["loss", "macro_dice", "learning rate"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.num_volumes
            )

        del sample
        del label
        del y_hat
        del result
        del labels
        del loss


class ClientCreator:
    def __init__(self, dbkey, redishost, volume_shape=[256] * 3):
        self.dbkey = dbkey
        self.redishost = redishost
        self.volume_shape = volume_shape
        self.subvolume_shape = None
        self.num_subcubes = None

    def set_shape(self, shape):
        self.subvolume_shape = shape
        self.coord_generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )

    def set_collection(self, collection):
        self.collection = collection

    def set_num_subcubes(self, num_subcubes):
        self.num_subcubes = num_subcubes

    def create_client(self, x):
        return create_client(
            x,
            redishost=self.redishost,
        )

    def mycollate(self, x):
        return collate_subcubes(
            x,
            self.coord_generator,
            samples=self.num_subcubes,
        )

    def mycollate_full(self, x):
        return mcollate(x)

    def mytransform(self, x):
        return mtransform(x)


def assert_equal_length(*args):
    assert all(
        len(arg) == len(args[0]) for arg in args
    ), "Not all parameter lists have the same length!"


if __name__ == "__main__":
    # hparams
    validation_percent = 0.1
    optimize_inline = False

    model_channels = 21 
    model_label = "_ss"

    #model_path = f"./logs/tmp/curriculum_enmesh_{model_channels}channels_ss/model.last.pth"
    model_path = ""
    logdir = f"./logs/tmp/curriculum_enmesh_{model_channels}channels_ss/"
    wandb_project = f"curriculum_{model_channels}_ss"

    client_creator = ClientCreator(DBKEY, REDISHOST)

    # set up parameters of your experiment
    cubesizes = [32, 48, 64, 96, 128, 192, 256]
    numcubes = [64, 64, 32, 16, 8, 4, 1]
    numvolumes = [1] * 7
    weights = [0.5] * 2 + [1]*5  # weights for the 0-class
    collections = [DBKEY] * 7
    epochlengths = [10000] * 4 + [25000] * 2 + [50000]
    epochs = [5] * 4 + [2]*2 + [6]
    prefetches = [24] * 7
    attenuates = [1] * 7

    assert_equal_length(
        cubesizes,
        numcubes,
        numvolumes,
        weights,
        collections,
        epochlengths,
        epochs,
        prefetches,
        attenuates,
    )

    start_experiment = 0
    for experiment in range(len(cubesizes)):
        COLLECTION = collections[experiment]
        num_subcubes = numcubes[experiment]
        num_volumes = numvolumes[experiment]
        num_epochs = epochs[experiment]

        off_brain_weight = weights[experiment]
        subvolume_shape = [cubesizes[experiment]] * 3
        onecycle_lr = rmsprop_lr = (
            attenuates[experiment]
            * 0.1
            * 8  # number of GPUs
            * num_subcubes
            * num_volumes
            / 256
        )
        epoch_length = epochlengths[experiment]
        n_fetch = prefetches[experiment]
        wandb_experiment = (
            f"{start_experiment + experiment:02} cube "
            + str(subvolume_shape[0])
            + " "
            + COLLECTION
            + model_label
        )

        # Set database parameters
        client_creator.set_collection(COLLECTION)
        client_creator.set_num_subcubes(num_subcubes)
        client_creator.set_shape(subvolume_shape)

        print("num epochs:", num_epochs)

        runner = CustomRunner(
            logdir=logdir,
            wandb_project=wandb_project,
            wandb_experiment=wandb_experiment,
            model_path=model_path,
            n_channels=model_channels,
            n_classes=n_classes,
            optimize_inline=optimize_inline,
            validation_percent=validation_percent,
            onecycle_lr=onecycle_lr,
            rmsprop_lr=rmsprop_lr,
            num_subcubes=num_subcubes,
            num_volumes=num_volumes,
            epoch_length=epoch_length,
            n_epochs=num_epochs,
            client_creator=client_creator,
            off_brain_weight=off_brain_weight,
            prefetches=n_fetch,
            db_name=COLLECTION,
            subvolume_shape=subvolume_shape,
        )
        runner.run()

        shutil.copy(
            logdir + "/model.last.pth",
            logdir + "/model.last." + str(subvolume_shape[0]) + ".pth",
        )

        model_path = logdir + "model.last.pth"

