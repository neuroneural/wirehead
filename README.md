<h1>wirehead</h1>

<div style="text-align:center; height:200px; overflow:hidden;">
  <img src="assets/wirehead_oct6.jpeg" alt="Wirehead" style="width:500px; object-fit:crop;">
</div>

---

<h1>Usage</h1>

* Example usage can be found in /src/dataloader.py

```
import wirehead as wh

tdataset = wh.whDataloader(
    transform=my_transform,      # User defined transformations 
    host=hostname,               # Hostname currently running wirehead's backend redis server 
    num_samples = sample_count)  # Number of samples to pull from wirehead
```
---

<h1>Description</h1>

* A dynamic data caching platform for low throughput synthetic data generation pipelines
* Built for SynthSeg on ARCTIC Slurm cluster
* Built on Redis for extremely high throughput and for funky database manipulation techniques

---

<h1>How it works</h1>

* Wirehead has 3 main components:
- The backend server, which hosts the redis server and the server management logic
- The backend generators, which use SynthSeg to create synthetic data, does preprocessing, and sends it off to the server
- The frontend dataloader, which only reads from the backend server

* The backend of wirehead is a redis server with 2 caches - 'db0', which is always full, and 'db1', which is always getting filled with fresh samples
* Starting wirehead's backend will flood 'db1' with generated samples. Once 'db1' is full, the key for the databases will be swapped, and it becomes 'db0'
* The frontend dataloader will detect when this happens, and start serving samples for your training job. The samples pulled from db0 in a loop, going index by index
* Wirehead's backend generation jobs can be scaled with as many nodes as one wishes, and the backend can be hosted off of infiniband for higher throughput

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
```
The training setup has custom modifications to Catalyst, so you'll have to copy those in manually
```
cp /src/utils/torch.py <your_conda>/envs/torch2/lib/python3.9/site-packages/catalyst/utils
```

Creating the SynthSeg environment. Follow detailed instructions from the [SynthSeg repo](https://github.com/BBillot/SynthSeg) to install it as per your system setup.

