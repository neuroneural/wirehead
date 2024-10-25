import numpy as np
from wirehead import WireheadGenerator 

def create_generator(n=4):
    # img = np.random.rand(256,256,256)
    # lab = np.random.rand(256,256,256)
    data = [np.random.rand(256) for i in range(n)]
    for i in range(100):
        yield tuple(data)

if __name__ == "__main__":
    brain_generator     = create_generator()
    wirehead_runtime    = WireheadGenerator(
        generator = brain_generator,
        config_path = "config.yaml",
        n_samples = 100
    )
    wirehead_runtime.run(verbose=True)
