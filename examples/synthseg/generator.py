import sys
import numpy as np
from wirehead import WireheadManager, WireheadGenerator
from time import time

import tensorflow as tf
from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing import preprocessing_pipe

WIREHEAD_CONFIG     = "config.yaml"
DATA_FILES          = [f"example.nii.gz"]

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import psutil
import os
import GPUtil

def get_memory_usage():
    # Get CPU memory usage
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # in MB

    # Get GPU memory usage
    gpus = GPUtil.getGPUs()
    gpu_mem = 0
    if gpus:
        gpu = gpus[0]  # Assuming you're using the first GPU
        gpu_mem = gpu.memoryUsed

    return cpu_mem, gpu_mem

# Get and print memory usage
cpu_usage, gpu_usage = get_memory_usage()
print(f"CPU Memory Usage: {cpu_usage:.2f} MB")
print(f"GPU Memory Usage: {gpu_usage:.2f} MB")

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
    start = time()
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())

# Get and print memory usage
        cpu_usage, gpu_usage = get_memory_usage()
        print(f"{(time()-start):.2f} CPU: {cpu_usage:.2f} MB. GPU: {gpu_usage:.2f} MB")
        yield (img, lab)

if __name__ == "__main__":
    brain_generator    = create_generator()
    wirehead_generator = WireheadGenerator(
        generator = brain_generator,
        config_path = WIREHEAD_CONFIG)
    wirehead_generator.run_generator()
