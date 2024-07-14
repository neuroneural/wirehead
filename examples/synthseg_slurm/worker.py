import os
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
        yields : tuple ( data0, data1, ...) """
    training_seg = DATA_FILES[worker_id%len(DATA_FILES)]
    brain_generator = BrainGenerator(
        DATA_FILES[training_seg],
        randomise_res=False,
    )
    print(f"Generator: SynthSeg is using {training_seg}",flush=True,)
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        yield (img, lab)

def my_task_id():
    """ Returns slurm task id """
    task_id = os.getenv(
        "SLURM_ARRAY_TASK_ID", "0"
    )
    return int(task_id)

if __name__ == "__main__":
    if my_task_id() == 0 and WIREHEAD_CONFIG != "":
        """ Becomes manager if first job on slurm """
        wirehead_manager = WireheadManager(
            config_path = WIREHEAD_CONFIG)
        wirehead_manager.run_manager()
    else:
        """ Otherwise, becomes generator """
        brain_generator    = create_generator(my_task_id())
        wirehead_generator = WireheadGenerator(
            generator = brain_generator,
            config_path = WIREHEAD_CONFIG)
        wirehead_generator.run_generator()
