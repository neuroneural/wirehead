import sys
import numpy as np
from wirehead import WireheadManager, WireheadGenerator

# Synthseg config
WIREHEAD_CONFIG     = "config.yaml"
PATH_TO_DATA        = "./SynthSeg/data/training_label_maps/"
DATA_FILES          = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
PATH_TO_SYNTHSEG    = './SynthSeg'

def hardware_setup():
    """ Clean slate to set up your hardware, ignore if none are needed """
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    sys.path.append(PATH_TO_SYNTHSEG)
    pass

# Create a generator function that yields desired samples
def create_generator(n_samples = 1000, task_id = 0, training_seg=None):
    """ Creates an iterator that returns data for mongo.
        Should contain all the dependencies of the brain generator
        Preprocessing should be applied at this phase 
        yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str)) """

    # 0. Optionally set up hardware configs
    hardware_setup()
    # 1. Declare your generator and its dependencies here
    sys.path.append(PATH_TO_SYNTHSEG)
    from SynthSeg.brain_generator import BrainGenerator
    training_seg = DATA_FILES[task_id % len(DATA_FILES)] if training_seg == None else training_seg
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(f"Generator: SynthSeg is generating off {training_seg}",flush=True,)
    # 2. Run your generator in a loop, and pass in your preprocessing options
    for i in range(n_samples):
        img, lab = brain_generator.generate_brain()
        print(f"Generator: Unique sample {i}")
        # 3. Yield your data, which will automatically be pushed to mongo
        yield (img, lab)

if __name__ == "__main__":
    brain_generator    = create_generator()
    wirehead_generator = WireheadGenerator(generator = brain_generator, config_path = WIREHEAD_CONFIG)
    wirehead_generator.run_generator()
