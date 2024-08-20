import gc
import os
from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing import preprocessing_pipe
from wirehead import WireheadGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_AUTOGRAPH_VERBOSITY"] = "0"

import tensorflow as tf

WIREHEAD_CONFIG = "config.yaml"
DATA_FILES = ["example.nii.gz"]
NUM_GENERATORS = 1

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception :
    pass


def create_generator(file_id=0):
    """Creates an iterator that returns data for mongo.
    Should contain all the dependencies of the brain generator
    Preprocessing should be applied at this phase
    yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str))
    """
    training_seg = DATA_FILES[file_id]
    brain_generator = BrainGenerator(
        training_seg,
    )
    print(f"Generator {file_id}: SynthSeg is using {training_seg}", flush=True)
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        yield (img, lab)
        gc.collect()


if __name__ == "__main__":
    brain_generator = create_generator()
    wirehead_generator = WireheadGenerator(
        generator=brain_generator, config_path=WIREHEAD_CONFIG
    )
    wirehead_generator.run(verbose=True)
