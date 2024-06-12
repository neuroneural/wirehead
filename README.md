# wirehead #

Caching system for horizontal scaling of synthetic data generators using MongoDB

---

## Usage ## 

See examples/unit for a minimal example 

Manager:
```
from wirehead import WireheadManager

if __name__ == "__main__":
    wirehead_runtime = WireheadManager(config_path="config.yaml")
    wirehead_runtime.run_manager()
```

Generator:

```
import numpy as np
from wirehead import WireheadGenerator 

def create_generator():
    while True: 
        img = np.random.rand(256,256,256)
        lab = np.random.rand(256,256,256)
        yield (img, lab)

if __name__ == "__main__":
    brain_generator     = create_generator()
    wirehead_runtime    = WireheadGenerator(
        generator = brain_generator,
        config_path = "config.yaml" 
    )
    wirehead_runtime.run_generator()
```

Dataset:
```
import torch
from wirehead import MongoheadDataset

dataset = MongoheadDataset(config_path = "config.yaml")

idx = [0] 
data = dataset[idx]
sample, label = data[0]['input'], data[0]['label']
```

## Installation 

For hosting MongoDB, refer to the official documentation
```
https://www.mongodb.com/docs/manual/installation/
```

Python environment setup

```
python3 -m venv wirehead 
pip install -e .
pip install -r requirements.txt
```

# TODO

- [ ] Clean up main wirehead files
- [ ] Documentation
  - Tutorial: how to make a generator, plug into wirehead, read from wirehead
  - Internals: what manager does, what generator does
  - Deeper: what each function in either object does
