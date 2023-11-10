from datetime import datetime
import threading
from typing import Dict
import os
import easybar
import shutil
import pickle
import sys
import redis


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

from mongoslabs.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)

# comment here
import wirehead as wh

# SEED = 0
# utils.set_global_seed(SEED)
# utils.prepare_cudnn(deterministic=False, benchmark=False) # crashes everything

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
os.environ["NCCL_P2P_LEVEL"] = "NVL"

volume_shape = [256] * 3
MAXSHAPE = 300

LABELNOW = ["sublabel", "gwmlabel", "50label"][0]
n_classes = [104, 3, 50][0]

MONGOHOST = "10.245.12.58"  # "arctrdcn018.rs.gsu.edu"
DBNAME = "MindfulTensors"
COLLECTION = "MRNslabs"
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
config_file = "modelAE.json"

# COLLECTION = "HCP"


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
        client_creator,
        off_brain_weight: float,
        prefetches=8,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        db_host=MONGOHOST,
        db_name=DBNAME,
        db_collection=COLLECTION,
    ):
        super().__init__()
        self._logdir = logdir
        self.model_path = model_path
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.optimize_inline = optimize_inline
        self.validation_percent = validation_percent
        self.onecycle_lr = onecycle_lr
        self.rmsprop_lr = rmsprop_lr
        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.prefetches = prefetches
        self.shape = subvolume_shape[0]
        self.batch_size = batch_size
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

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts = db[self.db_collection]
#        num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

# remember here
        # Wirehead code sub in####

        # Basic collate and transform functions that do pretty much nothing


        def mcollate(mlist, labelname="sublabel", cubesize=256):
            mdict = list2dict(mlist[0])
            data = []
            labels = []
            data = torch.empty(
                len(mdict), cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.float
            )
            labels = torch.empty(
                len(mdict), cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.long
            )
            cube = np.empty(shape=(cubesize, cubesize, cubesize))
            label = np.empty(shape=(cubesize, cubesize, cubesize))
            for i, subj in enumerate(mdict):
                for sub in mdict[subj]:
                    x, y, z = sub["coords"]
                    sz = sub["subdata"].shape[0]
                    cube[x : x + sz, y : y + sz, z : z + sz] = sub["subdata"]
                    label[x : x + sz, y : y + sz, z : z + sz] = sub[labelname]
                cube1 = preprocess_image(torch.from_numpy(cube).float())
                label1 = torch.from_numpy(label).long()
                data[i, :, :, :] = cube1
                labels[i, :, :, :] = label1
            del cube
            del label
            return data.unsqueeze(1), labels
        def rcollate(batch, cubesize=256):
            data = []
            labels = []
            data = torch.empty(
                len(batch), cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.float
            )
            labels = torch.empty(
                len(batch), cubesize, cubesize, cubesize, requires_grad=False, dtype=torch.long
            )

            items = batch[0] # Wirehead will only fetch with batchsize 1
            cube1 = torch.from_numpy(items[0]).float()
            label1 = torch.from_numpy(items[1]).long()
            data[0, :, :, :] = cube1
            labels[0, :, :, :] = label1
            return data.unsqueeze(1), labels


            
        def my_transform(x):
            return x
        def my_collate_fn(batch):
            item = batch[0]
            img = item[0]
            lab = item[1]
            # Add channel dimension (assuming single-channel data)
            img = torch.tensor(img)[None, ...]  # Shape becomes (1, 256, 256, 256)
            lab = torch.tensor(lab)[None, ...]  # Shape becomes (1, 256, 256, 256)
            # Stack along a new batch dimension
            batched_data = torch.stack([img, lab], dim=0)  # Shape becomes (2, 1, 256, 256, 256)
            return batched_data

        tdataset = wh.Dataloader(transform=my_transform, num_samples = 100) #modified

        tsampler = (
            MBatchSampler(tdataset, batch_size=1)
        )
        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                #sampler=tsampler,
                collate_fn=rcollate, #modifed
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"], #modified 
                persistent_workers=True,
                prefetch_factor=3,
                num_workers=3, #modified
            ),
            num_prefetches=12, #modified
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
        )
        # criterion = DiceLoss()
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


class ClientCreator:
    def __init__(self, dbname, mongohost, label, volume_shape=[256] * 3):
        self.dbname = dbname
        self.mongohost = mongohost
        self.volume_shape = volume_shape
        self.label = label
        self.subvolume_shape = None
        self.collection = None
        self.batch_size = None

    def set_shape(self, shape):
        self.subvolume_shape = shape
        self.coord_generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )

    def set_collection(self, collection):
        self.collection = collection

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def create_client(self, x):
        return create_client(
            x,
            dbname=self.dbname,
            colname=self.collection,
            mongohost=self.mongohost,
        )

    def mycollate(self, x):
        return collate_subcubes(
            x,
            self.coord_generator,
            labelname=self.label,
            samples=self.batch_size,
        )

    def mycollate_full(self, x):
        return mcollate(x, labelname=self.label)

    def mytransform(self, x):
        return mtransform(x, label=self.label)


def assert_equal_length(*args):
    assert all(
        len(arg) == len(args[0]) for arg in args
    ), "Not all parameter lists have the same length!"


if __name__ == "__main__":
    # hparams
    validation_percent = 0.1
    optimize_inline = False

    model_channels = 15
    model_label = "_startLARGE"

    model_path = f""
    logdir = f"./logs/tmp/curriculum_enmesh_{model_channels}channels_3_nodo/"
    wandb_project = f"curriculum_{model_channels}_sub"

    client_creator = ClientCreator(DBNAME, MONGOHOST, LABELNOW)

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

        # Set database parameters
        client_creator.set_collection(COLLECTION)
        client_creator.set_batch_size(batch_size)
        client_creator.set_shape(subvolume_shape)

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
            client_creator=client_creator,
            off_brain_weight=off_brain_weight,
            prefetches=n_fetch,
            db_collection=COLLECTION,
            subvolume_shape=subvolume_shape,
        )
        runner.run()

        shutil.copy(
            logdir + "/model.last.pth",
            logdir + "/model.last." + str(subvolume_shape[0]) + ".pth",
        )

        model_path = logdir + "model.last.pth"
