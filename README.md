# wirehead #

Caching system for horizontal scaling of synthetic data generators using Redis/MongoDB

---

## Usage ##

* Example usage can be found in /src/dataloader.py

```
import wirehead as wh

tdataset = wh.whDataloader(
    transform=my_transform,      # User defined transformations 
    host=hostname,               # Hostname currently running wirehead's backend redis server 
    num_samples = sample_count)  # Number of samples to pull from wirehead
```
---

## Description ##

* A dynamic data caching platform for low throughput synthetic data generation pipelines
* Built for SynthSeg on ARCTIC Slurm cluster
* Built on Redis for extremely high throughput and for funky database manipulation techniques

---
## Instructions ##

Create training environment
```
conda create wirehead_train python=3.9
conda activate wireheaed_train
pip3 install -U catalyst
conda install -c anaconda nccl
pip install redis
pip3 install pynvml
pip3 install scipy
pip3 install wandb
```
The training setup has custom modifications to Catalyst, so you'll have to copy those in manually
```
cp /src/utils/torch.py <your_conda>/envs/torch2/lib/python3.9/site-packages/catalyst/utils
```

Creating the SynthSeg environment. Follow detailed instructions from the [SynthSeg repo](https://github.com/BBillot/SynthSeg) to install it as per your system setup.

