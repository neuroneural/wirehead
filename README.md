# wirehead #

Caching system for horizontal scaling of synthetic data generators using MongoDB

---

## Usage ## 

See examples/unit for a minimal example 

```
import numpy as np
from wirehead import Runtime 

WIREHEAD_CONFIG = "config.yaml"

def create_generator():
    while True: 
        img = np.random.rand(256,256,256)
        lab = np.random.rand(256,256,256)
        yield (img, lab)

generator = create_generator()
wirehead_runtime    = Runtime(
    generator = generator,  # Specify generator 
    config_path = WIREHEAD_CONFIG # Specify config
)
```

Then, to run the generator, simply do 

```
wirehead_runtime.run_generator()
```

Or, to run the database manager,

```
wirehead_runtime.run_manager()
```

## MongoDB installation 

```
https://www.mongodb.com/docs/manual/installation/
```

# TODO

- [x] Split Runtime into Manager.py and Generator.py
- [ ] Merge into synthseg 





- [ ] Documentation
  - Tutorial: how to make a generator, plug into wirehead, read from wirehead
  - Internals: what manager does, what generator does
  - Deeper: what each function in either object does
- [ ] Split manager and generator into two files?
- [ ] Dump manager lock behavior into manager class?
