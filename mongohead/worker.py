import io
import os
import sys
import time
import bson
import torch
import threading
import numpy as np
from pymongo import MongoClient, ReturnDocument, ASCENDING

### User land ###

DBNAME              = "wirehead_mike"
COLLECTIONw         = "write.bin"
COLLECTIONr         = "read.bin"
COLLECTIONt         = "temp.bin"
COLLECTIONc         = "counters"
MONGOHOST           = "arctrdcn018.rs.gsu.edu"
PATH_TO_WIREHEAD    = "/data/users1/mdoan4/wirehead/"
PATH_TO_DATA        = (PATH_TO_WIREHEAD + "dependencies/synthseg/data/training_label_maps/")
DATA_FILES          = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
PATH_TO_SYNTHSEG    = '/data/users1/mdoan4/wirehead/dependencies/synthseg'
CHUNKSIZE           = 10
TARGET_COUNTER_VALUE= 5000 # Example threshold value

client              = MongoClient("mongodb://" + MONGOHOST + ":27017")
db                  = client[DBNAME]

LOG_METRICS         = True
BENCHMARK_SWAP      = True
EXPERIMENT_KIND     = 'mongohead'
COLLECTIONm         = 'metrics'
EXPERIMENT_NAME     = '2024-03-19_training'
WORKER_COUNT        = 20
LOCAL_RUNNING       = True

LABEL_MAP = np.asarray(
    [0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10]
    + [11, 12, 13, 14, 15]
    + [0] * 6
    + [1, 16, 0, 17]
    + [0] * 12
    + [18, 19, 20, 21, 0, 22, 23]
    + [0, 24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31]
    + [0] * 75
    + [3, 4]
    + [0] * 25
    + [20, 21]
    + [0] * 366,
    dtype="int",
).astype(np.uint8)


def create_generator(task_id, training_seg=None):
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
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())

        # 3. Yield your data, which will automatically be pushed to mongo
        yield ((img, lab), ('data', 'label'))

def preprocessing_pipe(data):
    """ Set up your preprocessing options here, ignore if none are needed """
    img, lab = data
    img = preprocess_image_min_max(img) * 255
    img = img.astype(np.uint8)
    lab = preprocess_label(lab)

    '''
    img_tensor = tensor2bin(torch.from_numpy(img))
    lab_tensor = tensor2bin(torch.from_numpy(lab))
    # TODO: Move these out of userland
    '''
    return (img, lab) 

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

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab.astype(np.uint8)]


def preprocess_image_min_max(img: np.ndarray) -> np.ndarray:
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img

### End of user land ###

def generate_and_insert(
    generator, collection_bin, counter_collection, chunk_size
):
    """ Preprocesses each sample from a generator and pushes it to mongodb """
    # 0. Fetch data from generator
    data = next(generator)
    # 1. Get the correct index for this current sample
    index = get_current_idx(counter_collection)
    # 2. Turn the data into a list of serialized chunks  
    chunks = chunkify(data, index, chunk_size)
    # 3. Push to mongodb + error handling
    push_chunks(collection_bin, chunks)

def chunkify(data, index, chunk_size):
    """ Converts a tuple of tensors and their labels into 
        a list of chunks of serialized objects for mongodb """
    def chunk_binobj(tensor_compressed, id, kind, chunksize):
        """ Convert chunksize from megabytes to bytes """
        chunksize_bytes = chunksize * 1024 * 1024

        # Calculate the number of chunks
        num_chunks = len(tensor_compressed) // chunksize_bytes
        if len(tensor_compressed) % chunksize_bytes != 0:
            num_chunks += 1

        # Yield chunks
        for i in range(num_chunks):
            start = i * chunksize_bytes
            end = min((i + 1) * chunksize_bytes, len(tensor_compressed))
            chunk = tensor_compressed[start:end]
            yield {
                "id": id,
                "chunk_id": i,
                "kind": kind,
                "chunk": bson.Binary(chunk),
            }
    chunks = []
    binobj, kinds = data
    for i, kind in enumerate(kinds):
        chunks += list(
            chunk_binobj(tensor2bin(torch.from_numpy(binobj[i])),
                         index,
                         kind,
                         chunk_size)) 
    return chunks

def initialize_generator():
    """ Initializes and runs a SynthSeg brain generator in a loop,
        preprocesses, then pushes to mongoDB"""
    print(
        "".join(["-"] * 50)
        + "\nI am a worker "
        + "\U0001F41D"
        + ", generating and inserting data.",
        flush=True,
    )

    brain_generator = create_generator(my_task_id())
    while True:
        print('ding')
        generate_and_insert(
            brain_generator, db[COLLECTIONw], db[COLLECTIONc], CHUNKSIZE
        )

def my_task_id() -> int:
    """ Returns slurm task id """
    task_id = os.getenv(
        "SLURM_ARRAY_TASK_ID", "0"
    )  # Default to '0' if not running under Slurm
    return int(task_id)

def tensor2bin(tensor):
    """ Serializes a torch tensor into a serialized IO buffer """ 
    # Flatten tensor to 1D
    # tensor_1d = tensor.flatten()
    # tensor_1d = tensor.flatten().to(torch.uint8)
    tensor_1d = tensor.to(torch.uint8)

    # Serialize tensor and get binary
    buffer = io.BytesIO()
    torch.save(tensor_1d, buffer)
    tensor_binary = buffer.getvalue()

    return tensor_binary


def get_current_idx(counter_collection):
    counter_doc = counter_collection.find_one_and_update(
        {"_id": "uniqueFieldCounter"},
        {"$inc": {"sequence_value": 1}},
        return_document=ReturnDocument.BEFORE,
    )
    return counter_doc["sequence_value"]

def push_chunks(collection_bin, chunks):
    try:
        collection_bin.insert_many(chunks)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        print(f"I expect you are renaming the collection", flush=True)
        time.sleep(1)


def assert_sequence(collection, TARGET_COUNTER_VALUE):
    # 1. Check the count of unique 'id' values
    unique_ids_count = len(collection.distinct("id"))
    assert (
        unique_ids_count == TARGET_COUNTER_VALUE
    ), f"Expected {TARGET_COUNTER_VALUE} unique ids, found {unique_ids_count}"

    # 2. Check that ids cover the range from 0 to TARGET_COUNTER_VALUE - 1
    # Create a set of all ids that should exist
    expected_ids_set = set(range(TARGET_COUNTER_VALUE))

    # Retrieve the set of unique ids from the collection
    actual_ids_set = set(collection.distinct("id"))

    # Check if the sets are equal
    assert (
        expected_ids_set == actual_ids_set
    ), "The 'id' values do not form a continuous sequence from 0 to TARGET_COUNTER_VALUE - 1"

    print(
        f"Assertion passed: 'id' values form a continuous sequence from 0 to {TARGET_COUNTER_VALUE - 1}.",
        flush=True,
    )


def reset_counter_and_collection(write_collection, counter_collection):
    # Delete all documents in the main collection that have creeped in
    # between the renaming and now. This operation is within a
    # transaction
    write_collection.delete_many({})
    # Reset the counter to zero
    result = counter_collection.update_one(
        {"_id": "uniqueFieldCounter"},  # Query part: the document to match
        {
            "$set": {"sequence_value": 0}
        },  # Update part: what to set if the document is matched/found
        upsert=True,  # This ensures that if the document doesn't exist, it will be inserted
    )
    # Delete all documents in the main collection that have creeped in
    # between the renaming and now. This operation is within a
    # transaction
    write_collection.delete_many({})
    write_collection.create_index([("id", ASCENDING)], background=True)

def log_metrics(generated):
    """ Pushes generator metrics onto database """
    experiment_name     = EXPERIMENT_NAME 
    kind                = "manager"
    curr_time           = time.time()
    total_samples       = generated
    metrics_collection  = db[COLLECTIONm]
    worker_count        = WORKER_COUNT
    # Create a BSON document with the metrics
    metrics_doc = {
        "experiment_name": experiment_name,
        "kind": kind,
        "timestamp": curr_time,
        "total_samples": total_samples,
        "worker_count": worker_count
    }
    # Insert the metrics document into the metrics collection
    metrics_collection.insert_one(metrics_doc)

# Function to watch the counter and perform actions when a threshold is reached
def watch_and_swap(TARGET_COUNTER_VALUE, generated, LOG_METRICS=LOG_METRICS):
    """ Watches the mongodb write collection's distinct id count
    When TARGET_COUNTER_VALUE is reached, swap() read with write"""
    def swap(generated):
        # Actions to be taken when the threshold is reached
        # Renaming the collection and creating a new one
        time.sleep(2)
        generated += TARGET_COUNTER_VALUE
        print(f"Generated samples so far {generated}", flush=True)
        db[COLLECTIONw].rename(COLLECTIONt, dropTarget=True)
        # Now atomically reset the counter to 0 and delete whatever records
        # may have been written between the execution of the previous line
        # and the next
        reset_counter_and_collection(
            db[COLLECTIONw], db[COLLECTIONc]
        )  # this is atomic
        # index temp.bin collection on id
        # db[COLLECTIONt].create_index([("id", ASCENDING)])
        # delete all records with id > (TARGET_COUNTER_VALUE - 1)
        result = db[COLLECTIONt].delete_many(
            {"id": {"$gt": TARGET_COUNTER_VALUE - 1}}
        )
        # Print the result of the deletion
        print(f"Documents deleted: {result.deleted_count}", flush=True)
        assert_sequence(db[COLLECTIONt], TARGET_COUNTER_VALUE)
        db[COLLECTIONt].rename(COLLECTIONr, dropTarget=True)

        # Log metrics on swap
        if LOG_METRICS: log_metrics(generated)
        return generated

    counter_doc = db[COLLECTIONc].find_one({"_id": "uniqueFieldCounter"})
    # print("checked the counter ", counter_doc["sequence_value"], flush=True)
    if counter_doc["sequence_value"] >= TARGET_COUNTER_VALUE:
        return swap(generated)
        
    return generated


   
def initialze_manager():
    """ Initializes the database manager, 
        swaps the mongo collections whenever TARGET_COUNTER_VALUE is hit. """
    print(
        "".join(["-"] * 50)
        + "\nI am the manager "
        + "\U0001F468\U0001F4BC"
        + ", watching the bottomline.", 
        flush=True,
    )
    reset_counter_and_collection(db[COLLECTIONw], db[COLLECTIONc])
    generated = 0
    while True:
        generated = watch_and_swap(TARGET_COUNTER_VALUE, generated, LOG_METRICS=LOG_METRICS)

def main():
    """ Starts the manager if node is the first job on slurm, also start the generator """
    """
    if my_task_id() == 0: # Checks if the job is the first job on slurm
        first_job_thread = threading.Thread(target=initialze_manager, args=())
        first_job_thread.start()
    """
    generator_thread = threading.Thread(target=initialize_generator, args=())
    generator_thread.start()
    """
    if my_task_id() == 0:
        first_job_thread.join()
    generator_thread.join()
    """

if __name__ == "__main__":
    main()
