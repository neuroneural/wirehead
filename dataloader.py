from datetime import datetime 
import os
import easybar

from torch.cuda.amp import autocast, GradScaler

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.dl import DataParallelEngine, DistributedDataParallelEngine

import nibabel as nib
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset,DistributedSampler


from dice import faster_dice
from meshnet import MeshNet
from mongoslabs.gencoords import CoordsGenerator
from blendbatchnorm import fuse_bn_recursively


from mongoslabs.mongoloader import (
        create_client,
        collate_subcubes,
        mcollate,
        MBatchSampler,
        MongoDataset,
        MongoClient,
        mtransform,
)


# Wirehead imports
import wirehead as wh

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


volume_shape = [256]*3
subvolume_shape = [256]*3


# All this stuff is probably useless for me
LABELNOW=["sublabel", "gwmlabel", "50label"][0]
MONGOHOST = "arctrdcn018.rs.gsu.edu"
DBNAME = 'MindfulTensors'
COLLECTION = 'MRNslabs'
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
config_file = "modelAE.json"
model_channels = 21
#coord_generator = CoordsGenerator(volume_shape, subvolume_shape)
model_label = "manual"
batched_subjs = 1
batch_size = 1
n_classes = 104
image_path = "/data/users2/splis/data/enmesh2/data/t1_c.nii.gz"


# Temp functions
def my_transform(x):
    return x
def my_collate_fn(batch):
    # Wirehead always fetches with batch = 1
    item = batch[0]
    img = item[0] 
    lab = item[1] 
    return torch.tensor(img), torch.tensor(lab)

# Dataloading with wirehead 
tdataset = wh.wirehead_dataloader_v3(transform=my_transform, num_samples = 10)
tsampler= (
        MBatchSampler(tdataset)
        )
tdataloader = BatchPrefetchLoaderWrapper(
        DataLoader(
            tdataset,
            #sampler=tsampler,
            collate_fn = my_collate_fn,
            pin_memory=True,
            #worker_init_fn=create_client,
            num_workers=1,
            ),
        num_prefetches=1 
        )
for loader in [tdataloader]:
    for i, batch in enumerate(loader):
        output_dir = "../samples/"
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img, lab = batch
        img_np = img.cpu().numpy()
        lab_np = lab.cpu().numpy()
        # Create Nifti images
        img_nii = nib.Nifti1Image(img_np, np.eye(4))
        lab_nii = nib.Nifti1Image(lab_np, np.eye(4))

        # Save in samples directory
        nib.save(img_nii, os.path.join(output_dir, f"batch_{i}_img.nii.gz"))
        nib.save(lab_nii, os.path.join(output_dir, f"batch_{i}_lab.nii.gz"))

        easybar.print_progress(i, len(loader))

print("hi")

