# This is a simplified rewrite to work with wirehead
# for original implmeentation, please reference
# curriculum_training_sub.py

import sys
sys.path.append('/data/users1/mdoan4/wirehead/src')
sys.path.append('/data/users1/mdoan4/wirehead/src/utils')
import wirehead as wh

from datetime import datetime
import os
import sys
import redis
import shutil

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
os.environ["NCCL_P2P_LEVEL"] = "NVL"

volume_shape = [256] * 3
MAXSHAPE = 300

n_classes = [104, 3, 50][0]

WIREHEAD_HOST = "arctrdagn019"  
WIREHEAD_PORT =  6379
WIREHEAD_NUMSAMPLES = 100 # specifies how many samples to fetch from wirehead 

config_file = "modelAE.json"

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
            n_epochs: int,
            optimize_inline: bool,
            validation_percent: float,
            onecycle_lr: float,
            rmsprop_lr: float,
            batch_size: int,
            off_brain_weight: float,
            prefetches=8,
            volume_shape=[256] * 3,
            subvolume_shape=[256] * 3,
            db_host=WIREHEAD_HOST,
            db_port=WIREHEAD_PORT,
    ):
        super().__init__()
        self._logdir = logdir
        self._logdir = logdir
        self.model_path = model_path
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.optimize_inline = optimize_inline
        self.validation_percent = validation_percent
        self.onecycle_lr = onecycle_lr # what is this
        self.rmsprop_lr = rmsprop_lr # what is this
        self.db_host = db_host
        self.db_port= db_port
        self.prefetches = prefetches
        self.shape = subvolume_shape[0]
        self.batch_size = batch_size
        self.off_brain_weight = off_brain_weight
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
            "wandb": dl.WandbLogger(
                project=self.wandb_project,
                name=self.wandb_experiment,
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
        return self.n_epochs   
    
    def get_loaders(self):
        # 'r'functions are just functions designed to work with wirehead
        def rcollate(batch, size=256): # 'size' is just 'cubesize'
            data = torch.empty(len(batch), size, size, size, requires_grad=False, dtype=torch.float)
            labels = torch.empty(len(batch), size, size, size, requires_grad=False, dtype=torch.long)
            items = batch[0] # Wirehead will only fetch with batchsize 1
            data[0, :, :, :] = torch.from_numpy(items[0]).float()
            labels[0, :, :, :] = torch.from_numpy(items[1]).long()
            return data.unsqueeze(1), labels

        def rtransform(x):
            return x

        tdataset = wh.Dataloader(host=self.db_host,
                                 port=self.db_port,
                                 transform=rtransform,
                                 num_samples=WIREHEAD_NUMSAMPLES)
        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                collate_fn=rcollate,
                pin_memory=True,
                persistent_workers=True,
                num_workers=3,
            ),
            num_prefetches=12,
        )
        return {"train": tdataloader}
    
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

    def get_criterion(self):
        class_weight = torch.FloatTensor(
            [self.off_brain_weight] + [1.0] * (self.n_classes - 1)
        ).to(self.engine.device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=0.01
        ) # so we're using cross entropy loss, interesting
        # criterion = DiceLoss()
        return criterion       

    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=100,
            pct_start=0.2,
            epochs=self.n_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )
        return scheduler

    def get_callbacks(self):
        checkpoint_params = {"save_best": True, "metric_key": "macro_dice"}
        if self.model_path:
            checkpoint_params = {"resume_model": self.model_path}
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    # these are catalyst functions
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
                self.batch_metrics[key].item(), self.batch_size
            )
        del sample
        del label
        del y_hat
        del result
        del labels
        del loss

def assert_equal_length(*args):
    assert all(
        len(arg) == len(args[0]) for arg in args
    ), "Not all parameter lists have the same length!"

if __name__ == "__main__":
    # hparams
    validation_percent = 0.1
    optimize_inline = False

    #TODO: write better file fetching
    model_channels = 15
    model_label = "_startLARGE"

    model_path = f""
    logdir = f"./logs/tmp/curriculum_enmesh_{model_channels}channels_3_nodo/"
    wandb_project = f"curriculum_{model_channels}_sub"

    # set up parameters of your experiment
    cubesizes = [256] * 6
    batchsize = [1] * 6
    weights = [0.5] * 2 + [1] * 4  # weights for the 0-class
    collections = ["HCP", "MRNslabs"] * 3
    epochs = [50] * 2 + [100] * 2 + [50, 10]
    prefetches = [24] * 6
    attenuates = [1] * 6

    assert_equal_length(
        cubesizes,
        batchsize,
        weights,
        collections,
        epochs,
        prefetches,
        attenuates,
    )

    start_experiment = 0
    for experiment in range(len(cubesizes)):
        COLLECTION = collections[experiment]
        batch_size = batchsize[experiment]

        off_brain_weight = weights[experiment]
        subvolume_shape = [cubesizes[experiment]] * 3
        onecycle_lr = rmsprop_lr = (
            attenuates[experiment] * 0.1 * 8 * batch_size / 256
        )
        n_epochs = epochs[experiment]
        n_fetch = prefetches[experiment]
        wandb_experiment = (
            f"{start_experiment + experiment:02} cube "
            + str(subvolume_shape[0])
            + " "
            + COLLECTION
            + model_label
        )

        runner = CustomRunner(
            logdir=logdir,
            wandb_project=wandb_project,
            wandb_experiment=wandb_experiment,
            model_path=model_path,
            n_channels=model_channels,
            n_classes=n_classes,
            n_epochs=n_epochs,
            optimize_inline=optimize_inline,
            validation_percent=validation_percent,
            onecycle_lr=onecycle_lr,
            rmsprop_lr=rmsprop_lr,
            batch_size=batch_size,
            off_brain_weight=off_brain_weight,
            prefetches=n_fetch,
            subvolume_shape=subvolume_shape,
        )
        runner.run()

        shutil.copy(
            logdir + "/model.last.pth",
            logdir + "/model.last." + str(subvolume_shape[0]) + ".pth",
        )
        model_path = logdir + "model.last.pth"