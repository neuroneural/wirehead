from time import time
import gc
import tensorflow as tf
from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing import preprocessing_pipe
import multiprocessing
from queue import Empty
from wirehead import WireheadManager, WireheadGenerator

WIREHEAD_CONFIG = "config.yaml"
DATA_FILES = ["example.nii.gz"]
NUM_GENERATORS = 1

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

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
        # np.save here if you want to see your outputs
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

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    queue = multiprocessing.Queue()
    
    processes = []
    for i in range(NUM_GENERATORS):
        p = multiprocessing.Process(target=run_wirehead_generator, args=(i, queue))
        p.start()
        processes.append(p)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping generators...")
        for p in processes:
            p.terminate()
