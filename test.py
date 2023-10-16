import gc
import time

import cc3d
import nibabel as nib
import numpy as np
import torch

from blendbatchnorm import fuse_bn_recursively
from meshnet import MeshNet


def preprocess_image(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


volume_shape = [256, 256, 256]
subvolume_shape = [38, 38, 38]
n_subvolumes = 1024
n_classes = 3
atlas_classes = 104
n_classes = [atlas_classes, n_classes, 50][0]
connectivity = 26
image_path = "./data/"+["mprage003-c.nii.gz", "T1_baby.nii.gz", "t1_c.nii.gz"][0]
config_file = "modelAE.json"
model_channels = 48

model_path = "logs/tmp/enmesh_48channels_ELU/model.last.pth"

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

meshnet_model = MeshNet(in_channels=1,
                        n_classes=n_classes,
                        channels=model_channels,
                        config_file=config_file)

checkpoint = torch.load(model_path)
meshnet_model.load_state_dict(checkpoint)

# meshnet_model.load_state_dict(
#     torch.load(model_path, map_location=device)["model_state_dict"]
# )
meshnet_model.eval()

meshnet_model.to(device)
mnm = fuse_bn_recursively(meshnet_model)
del meshnet_model
mnm.model.eval()

layers = [p for p in mnm.model.parameters()]
# for p in mnm.model.parameters():

for p in layers:
    p.grad = None
    p.requires_grad = False
print("starting")

img1 = nib.load(image_path)
img = np.asanyarray(img1.dataobj)
img = torch.from_numpy(img).to(device).float()
img = preprocess_image(img)
img = torch.stack([img], dim=0).unsqueeze(1)


t0 = time.time()
input = img
# with torch.no_grad():
for layer in mnm.model:
    input = layer(input)
t1 = time.time()
print(t1 - t0)

result = np.squeeze(torch.argmax(torch.squeeze(input), 0).cpu().numpy()).astype(np.uint8)

# remove small disconnected clusters
rr = result > 0
r, N = cc3d.connected_components(rr, connectivity=connectivity, return_N=True)
cluster_sizes = []
for segid in range(1, N + 1):
    extracted_image = r == segid
    cluster_sizes.append(np.sum(extracted_image))
result = (result * (r == np.argmax(cluster_sizes) + 1)).astype(np.uint8)

# save predicted labels to nifti
res_nii = nib.Nifti1Image(result, img1.affine, img1.header)
nib.save(res_nii, "result.nii")
