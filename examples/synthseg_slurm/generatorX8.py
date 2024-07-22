from time import time
import gc
import tensorflow as tf
from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing import preprocessing_pipe
import psutil
import os
import GPUtil
import csv
import multiprocessing
from queue import Empty
from wirehead import WireheadManager, WireheadGenerator

WIREHEAD_CONFIG = "config.yaml"
DATA_FILES = ["example.nii.gz"]
CSV_OUTPUT_FILE = "whmemory_usage_log.csv"
NUM_GENERATORS = 8

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
    """ Creates an iterator that returns data for mongo.
        Should contain all the dependencies of the brain generator
        Preprocessing should be applied at this phase 
        yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str)) """
    training_seg = DATA_FILES[worker_id % len(DATA_FILES)]
    brain_generator = BrainGenerator(
        training_seg,
        randomise_res=False,
    )
    print(f"Generator {worker_id}: SynthSeg is using {training_seg}", flush=True)
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        yield (img, lab)
        gc.collect()

def run_wirehead_generator(worker_id, queue):
    brain_generator = create_generator(worker_id)
    wirehead_generator = WireheadGenerator(
        generator=brain_generator,
        config_path=WIREHEAD_CONFIG
    )
    
    for item in wirehead_generator.run_generator():
        queue.put((worker_id, time()))

def log_memory_usage(queue):
    start_time = time()
    with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Time (s)', 'CPU Memory (MB)', 'GPU Memory (MB)', 'Generator ID'])
        
        while True:
            try:
                worker_id, gen_time = queue.get(timeout=1)
                cpu_usage, gpu_usage = get_memory_usage()
                elapsed_time = gen_time - start_time
                
                csvwriter.writerow([f"{elapsed_time:.2f}", f"{cpu_usage:.2f}", f"{gpu_usage:.2f}", worker_id])
                csvfile.flush()
                
                print(f"{elapsed_time:.2f} CPU: {cpu_usage:.2f} MB. GPU: {gpu_usage:.2f} MB (Generator {worker_id})")
            except Empty:
                pass

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    queue = multiprocessing.Queue()
    
    processes = []
    for i in range(NUM_GENERATORS):
        p = multiprocessing.Process(target=run_wirehead_generator, args=(i, queue))
        p.start()
        processes.append(p)
    
    # log_process = multiprocessing.Process(target=log_memory_usage, args=(queue,))
    # log_process.start()
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping generators...")
        for p in processes:
            p.terminate()
        # log_process.terminate()
