from datetime import datetime
import os
import easybar

from torch.cuda.amp import autocast, GradScaler

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.dl import DataParallelEngine, DistributedDataParallelEngine

import ipdb
import nibabel as nib
import numpy as np

import torch
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR, CosineAnnealingLR, ChainedScheduler, CyclicLR, StepLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dice import faster_dice
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

# SEED = 0
# utils.set_global_seed(SEED)
# utils.prepare_cudnn(deterministic=False, benchmark=False) # crashes everything

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['NCCL_IB_DISABLE'] = '0'
os.environ['NCCL_SOCKET_IFNAME'] = '^docker0'

volume_shape = [256]*3
subvolume_shape = [256]*3

LABELNOW=["sublabel", "gwmlabel", "50label"][0]
MONGOHOST = "arctrdcn018.rs.gsu.edu"
DBNAME = 'MindfulTensors'
COLLECTION = 'MRNslabs'
#COLLECTION = "HCP"
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
config_file = "modelAE.json"
model_channels = 48
coord_generator = CoordsGenerator(volume_shape, subvolume_shape)
model_label = "_warmup"

def createclient(x):
    return create_client(x, dbname=DBNAME,
                         colname=COLLECTION,
                         mongohost=MONGOHOST)

def mycollate_full(x):
    return mcollate(x, labelname=LABELNOW)

def mycollate(x):
    return collate_subcubes(x, coord_generator,
                            labelname=LABELNOW,
                            samples=1)
def mytransform(x):
    return mtransform(x, label=LABELNOW)

class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]


# CustomRunner – PyTorch for-loop decomposition
# https://github.com/catalyst-team/catalyst#minimal-examples
class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir: str,
        model_path: str,
        n_classes: int,
        n_epochs: int,
        optimize_inline: bool,
        validation_percent: float,
        onecycle_lr: float,
        rmsprop_lr: float
    ):
        super().__init__()
        self._logdir = logdir
        self.model_path = model_path
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.optimize_inline = optimize_inline
        self.validation_percent = validation_percent
        self.onecycle_lr = onecycle_lr
        self.rmsprop_lr = rmsprop_lr
        self.scaler = GradScaler()

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            return dl.DistributedDataParallelEngine(
                mixed_precision='fp16',
                #ddp_kwargs={"find_unused_parameters": True}
            )
        else:
            return dl.GPUEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            # "tensorboard": dl.TensorboardLogger(logdir=self._logdir,
            #                                     log_batch_metrics=True),
            "wandb": dl.WandbLogger(project=f"enmesh_{model_channels}channels",
                                    name="subcube "+str(subvolume_shape[0])+COLLECTION+model_label,
                                    log_batch_metrics=True)
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

        client = MongoClient("mongodb://" + MONGOHOST + ":27017")
        db = client[DBNAME]
        posts = db[COLLECTION]
        num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

        tdataset = MongoDataset(
            range(int((1 - self.validation_percent) * num_examples)),
            mytransform,
            None,
            id=INDEX_ID,
            fields=VIEWFIELDS,
        )
        tsampler = (
            DistributedSamplerWrapper(MBatchSampler(tdataset, batch_size=1))
            if self.engine.is_ddp
            else MBatchSampler(tdataset, batch_size=1)
        )
        # tsampler = mBatchSampler(tdataset, batch_size=1)
        #
        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                sampler=tsampler,
                collate_fn=mycollate_full,
                pin_memory=True,
                worker_init_fn=createclient,
                num_workers=8,
                ),
             num_prefetches=4
             )

        vdataset = MongoDataset(
            range(
                num_examples - int(self.validation_percent * num_examples), num_examples
            ),
            mytransform,
            None,
            id=INDEX_ID,
            fields=VIEWFIELDS,
        )
        vsampler = (
            DistributedSamplerWrapper(MBatchSampler(vdataset, batch_size=1))
            if self.engine.is_ddp
            else MBatchSampler(vdataset, batch_size=1)
        )
        # vsampler = mBatchSampler(vdataset, batch_size=1)
        # vdataloader =
        #vdataloader = BatchPrefetchLoaderWrapper(
        vdataloader = DataLoader(
            vdataset,
            sampler=vsampler,
            pin_memory=True,
            collate_fn=mycollate_full,
            worker_init_fn=createclient,
            num_workers=8,
            )
            # num_prefetches=16
            # )

        # for i in range(1):
        #     print(i, "burning")
        #     for loader in [tdataloader, vdataloader]:
        #         for i, (x, y) in enumerate(loader):
        #             easybar.print_progress(i, len(loader))
        #             x_, y_ = x.cuda(), y.cuda()

        # exit()
        return {"train": tdataloader, "valid": vdataloader}

    def get_model(self):
        # model = enMesh(in_channels=1,
        #     n_classes=self.n_classes,
        #     channels=model_channels,
        #     config_file=config_file,
        #     optimize_inline=self.optimize_inline
        #     )

        model = enMesh_checkpoint(in_channels=1,
            n_classes=self.n_classes,
            channels=model_channels,
            config_file=config_file)
        #model = torch.compile(model)
        # if len(self.model_path) > 0:
        #     checkpoint = torch.load(self.model_path)
        #     model.load_state_dict(checkpoint)
            # utils.unpack_checkpoint(
            #     checkpoint=utils.load_checkpoint(path=self.model_path), model=model
            # )
        return model

    def get_criterion(self):
        class_weight = torch.FloatTensor([1.2] + [1.0] * (self.n_classes - 1)).to(self.engine.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.01)
        return criterion

    def get_optimizer(self, model):
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.rmsprop_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        # cycles = 3
        # warpmup_portion = 0.3
        # base_percent = 0.05
        # cycle_iters = (len(self.loaders["train"])*self.n_epochs)/cycles
        # upcycle = int(warpmup_portion*cycle_iters)
        # postcycle = cycle_iters - upcycle
        # scheduler = CyclicLR(
        #     optimizer,
        #     base_lr = base_percent*self.onecycle_lr,
        #     max_lr = self.onecycle_lr,
        #     step_size_up = upcycle,
        #     gamma = 0.995,
        #     step_size_down = postcycle,
        #     mode = 'exp_range',
        #     cycle_momentum = False,
        #     last_epoch=-1
        #     )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            pct_start=0.2,
            #epochs=1,
            epochs=self.n_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )

        #scheduler = StepLR(optimizer, step_size=len(self.loaders["train"]), gamma=0.3)
                
        # warpmup_portion = 0.2
        # warmup_steps = int(warpmup_portion*len(self.loaders["train"])*self.n_epochs)
        # warmup_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)
        # annealing_scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=len(self.loaders["train"])*self.n_epochs,
        #     eta_min=0,
        #     last_epoch=-1
        # )
        # scheduler = ChainedScheduler([warmup_scheduler, annealing_scheduler])
        return scheduler

    def get_callbacks(self):
        checkpoint_params = {}
        if self.model_path:
            checkpoint_params = {'resume_model': self.model_path}
        return {
            "checkpoint": dl.CheckpointCallback(self._logdir, **checkpoint_params),
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
        # unpack the batche
        sample, label = batch
        # run model forward/backward pass
        if self.model.training:
            # loss, y_hat = self.model.forward(x=sample, y=label,
            #                                  loss=self.criterion,
            #                                  verbose=False)
            with autocast():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
            self.scaler.scale(loss).backward()
            # y_hat = self.model.forward(sample)
            # loss = self.criterion(y_hat, label)
            # loss.backward()
            if not self.optimize_inline:
                #self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            with torch.inference_mode():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
        with torch.inference_mode():
            result = torch.squeeze(torch.argmax(y_hat, 1)).long()
            labels = torch.squeeze(label)
            dice = torch.mean(faster_dice(result, labels, range(self.n_classes)))

        self.batch_metrics.update({"loss": loss, "macro_dice": dice,
                                   "learning rate": torch.tensor(self.optimizer.param_groups[0]['lr'])})

        for key in ["loss", "macro_dice", "learning rate"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        del sample
        del label
        del y_hat
        del result
        del labels
        del loss


if __name__ == "__main__":
    # hparams
    validation_percent = 0.1
    onecycle_lr = rmsprop_lr = 1/256
    n_classes = [104, 3, 50][0]
    n_epochs = 1
    optimize_inline = False
    model_path = f"./logs/tmp/enmesh_{model_channels}channels_ELU_largeLR/model.last.pth"
#    logdir = f"./logs/tmp/enmesh_"+COLLECTION+"subcube"+str(subvolume_shape[0])+f"_{datetime.utcnow().strftime('%y%m%d.%H%M%S')}"
    logdir = f"./logs/tmp/enmesh_{model_channels}channels_ELU/"

    runner = CustomRunner(
        logdir=logdir,
        model_path=model_path,
        n_classes=n_classes,
        n_epochs=n_epochs,
        optimize_inline=optimize_inline,
        validation_percent=validation_percent,
        onecycle_lr=onecycle_lr,
        rmsprop_lr=rmsprop_lr
        )
    runner.run()
