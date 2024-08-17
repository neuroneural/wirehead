import numpy as np
import time
from wirehead import WireheadGenerator 

def create_generator():
    img = np.random.rand(256,256,256)
    lab = np.random.rand(256,256,256)
    # img = np.random.rand(256)
    # lab = np.random.rand(256)
    while True:
        yield (img, lab)

if __name__ == "__main__":
    brain_generator     = create_generator()
    wirehead_runtime    = WireheadGenerator(
        generator = brain_generator,
        config_path = "config.yaml",
    )
    wirehead_runtime.run(verbose=True)
