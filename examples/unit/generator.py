import threading
import numpy as np
from wirehead import Runtime 

WIREHEAD_CONFIG = "config.yaml"

def create_generator():
    while True: 
        img = np.random.rand(256,256,256)
        lab = np.random.rand(256,256,256)
        yield (img, lab)

if __name__ == "__main__":
    # Plug into wirehead 
    brain_generator     = create_generator()
    wirehead_runtime    = Runtime(
        generator = brain_generator,  # Specify generator 
        config_path = WIREHEAD_CONFIG # Specify config
    )
    wirehead_runtime.run_generator()
