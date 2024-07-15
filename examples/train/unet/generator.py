import sys
import numpy as np
from wirehead import WireheadManager, WireheadGenerator

from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing import preprocessing_pipe

WIREHEAD_CONFIG     = "config.yaml"
DATA_FILES          = [f"example.nii.gz"]

def create_generator(worker_id=0):
    """ Creates an iterator that returns data for mongo.
        Should contain all the dependencies of the brain generator
        Preprocessing should be applied at this phase 
        yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str)) """
    training_seg = DATA_FILES[worker_id]
    brain_generator = BrainGenerator(
        DATA_FILES[0],
        randomise_res=False,
    )
    print(f"Generator: SynthSeg is using {training_seg}",flush=True,)
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        yield (img, lab)

if __name__ == "__main__":
    brain_generator    = create_generator()
    wirehead_generator = WireheadGenerator(
        generator = brain_generator,
        config_path = WIREHEAD_CONFIG)
    wirehead_generator.run_generator()
