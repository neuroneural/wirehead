from datetime import datetime
import os
from torch.cuda.amp import autocast, GradScaler
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.dl import DataParallelEngine, DistributedDataParallelEngine
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from dice import faster_dice, DiceLoss
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
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
volume_shape = [256]*3
subvolume_shape = [256]*3
LABELNOW=["sublabel", "gwmlabel", "50label"][0]
MONGOHOST = "arctrdcn018.rs.gsu.edu"
DBNAME = 'MindfulTensors'
COLLECTION = 'MRNslabs'
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
config_file = "modelAE.json"
model_channels = 21
coord_generator = CoordsGenerator(volume_shape, subvolume_shape)
model_label = "manual"
batched_subjs = 1
batch_size = 1
n_classes = 104
image_path = "/data/users2/splis/data/enmesh2/data/t1_c.nii.gz"
def createclient(x):
    return create_client(x, dbname=DBNAME,
                         colname=COLLECTION,
                         mongohost=MONGOHOST)
def mycollate_full(x):
    return mcollate(x, labelname=LABELNOW)
def mytransform(x):
    return mtransform(x, label=LABELNOW)
model_path = "/data/users2/splis/data/enmesh2/logs/tmp/enmesh_21channels_ELU_manual/model.last.pth"
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
meshnet_model = MeshNet(in_channels=1,
                        n_classes=n_classes,
                        channels=model_channels,
                        config_file=config_file)
checkpoint = torch.load(model_path)
meshnet_model.load_state_dict(checkpoint)
meshnet_model.eval()
meshnet_model.to(device)
mnm = fuse_bn_recursively(meshnet_model)
del meshnet_model
mnm.model.eval()
client = MongoClient("mongodb://" + MONGOHOST + ":27017")
db = client[DBNAME]
posts = db[COLLECTION]
num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)
tdataset = MongoDataset(
    range(num_examples),
    mytransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
    )
tsampler = (
    MBatchSampler(tdataset, batch_size=1)
    )
tdataloader = BatchPrefetchLoaderWrapper(
    DataLoader(
        tdataset,
        sampler=tsampler,
        collate_fn=mycollate_full,
        pin_memory=True,
        worker_init_fn=createclient,
        num_workers=8,
        ),
     num_prefetches=16
     )
img = nib.load(image_path)
saved = 0
for i in range(1):
    print(i, "burning")
    for loader in [tdataloader]:
        for i, (x, y) in enumerate(loader):
            #x_, y_ = x.to(device), y.to(device)
            input = x.detach().clone()
            for layer in mnm.model:
                input = layer(input)
            result = torch.squeeze(torch.argmax(input, 1)).long()
            labels = torch.squeeze(y)
            dice = torch.mean(faster_dice(result, labels, range(n_classes)))
            if dice < 0.1:
                print(dice, i)
                result = np.squeeze(result.cpu().numpy().astype(np.uint8))
                #result = np.squeeze(torch.argmax(torch.squeeze(input), 0).cpu().numpy()).astype(np.uint8)                
                label = np.squeeze(y.cpu().numpy().astype(np.uint8))
                data = np.squeeze(x.cpu().numpy())
                res_nii = nib.Nifti1Image(result, img.affine, img.header)
                label_nii = nib.Nifti1Image(label, img.affine, img.header)
                input_nii = nib.Nifti1Image(data, img.affine, img.header)
                nib.save(input_nii, f"{i}_input.nii")
                nib.save(res_nii, f"{i}_pred.nii")
                nib.save(label_nii, f"{i}_label.nii")
                saved += 1
