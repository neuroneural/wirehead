from time import time
import gc
import tensorflow as tf
from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing import preprocessing_pipe
import psutil
import os
import GPUtil
import csv

WIREHEAD_CONFIG = "config.yaml"
DATA_FILES = ["example.nii.gz"]
CSV_OUTPUT_FILE = "memory_usage_log.csv"

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # in MB
    gpus = GPUtil.getGPUs()
    gpu_mem = 0
    if gpus:
        gpu = gpus[0]  # Assuming you're using the first GPU
        gpu_mem = gpu.memoryUsed
    return cpu_mem, gpu_mem

def create_generator(worker_id=0):
    training_seg = DATA_FILES[worker_id]
    brain_generator = BrainGenerator(
        DATA_FILES[0],
        randomise_res=False,
    )
    print(f"Generator {str(worker_id)}: SynthSeg is using {training_seg}", flush=True)
    return brain_generator
    
# Open CSV file for writing
with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time (s)', 'CPU Memory (MB)', 'GPU Memory (MB)'])
    
    brain_generators = [create_generator() for i in range(8)] 

    start = time()
    while True:
        for brain_generator in brain_generators:
            img, lab = preprocessing_pipe(brain_generator.generate_brain())
            cpu_usage, gpu_usage = get_memory_usage()
            elapsed_time = time() - start
            
            # Write to CSV
            csvwriter.writerow([f"{elapsed_time:.2f}", f"{cpu_usage:.2f}", f"{gpu_usage:.2f}"])
            csvfile.flush()  # Ensure data is written immediately
            
            print(f"{elapsed_time:.2f} CPU: {cpu_usage:.2f} MB. GPU: {gpu_usage:.2f} MB")
            gc.collect()
