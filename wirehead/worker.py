from pymongo import MongoClient, ReturnDocument, ASCENDING
import os
import time
import numpy as np
import io
import torch
import bson
### Synthseg imports ###############################
import sys
# synthseg path
sys.path.append('/data/users1/mdoan4/wirehead/dependencies/synthseg') 
from SynthSeg.brain_generator import BrainGenerator
####################################################
from SynthSeg.brain_generator import BrainGenerator
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

DBNAME = "wirehead_mike"
COLLECTIONw = "write.bin"
COLLECTIONr = "read.bin"
COLLECTIONt = "temp.bin"
COLLECTIONc = "counters"
MONGOHOST = "arctrdcn018.rs.gsu.edu"
PATH_TO_WIREHEAD = "/data/users1/mdoan4/wirehead/"
PATH_TO_DATA = (
    PATH_TO_WIREHEAD + "dependencies/synthseg/data/training_label_maps/"
)
DATA_FILES = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
CHUNKSIZE = 10

TARGET_COUNTER_VALUE = 1000

client = MongoClient("mongodb://" + MONGOHOST + ":27017")
db = client[DBNAME]


###################################
###### Defaults for logging #######
LOG_METRICS = True
COLLECTIONm = 'metrics'
EXPERIMENT_NAME = '2024-03-18'
WORKER_COUNT = 20
LOCAL_RUNNING = True
###################################


LABEL_MAP = np.asarray(
    [0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10]
    + [11, 12, 13, 14, 15]
    + [0] * 6
    + [1, 16, 0, 17]
    + [0] * 12
    + [18, 19, 20, 21, 0, 22, 23]
    + [0, 24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31]
    + [0] * 75
    + [3, 4] + [0] * 25
    + [20, 21]
    + [0] * 366,
    dtype="int",
).astype(np.uint8)


def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab.astype(np.uint8)]


def preprocess_image_min_max(img: np.ndarray) -> np.ndarray:
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img


def my_task_id():
    task_id = os.getenv(
        "SLURM_ARRAY_TASK_ID", "0"
    )  # Default to '0' if not running under Slurm
    return int(task_id)


# Function to check if this is the first job based on SLURM_ARRAY_TASK_ID
def is_first_job():
    return my_task_id() == 0


def create_generator(task_id, training_seg=None):
    if training_seg == None:
        training_seg = DATA_FILES[task_id % len(DATA_FILES)]
    else:
        training_seg = training_seg
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(f"Generator: SynthSeg is generating off {training_seg}", flush=True)
    return brain_generator


def tensor2bin(tensor):
    # Flatten tensor to 1D
    # tensor_1d = tensor.flatten()
    # tensor_1d = tensor.flatten().to(torch.uint8)
    tensor_1d = tensor.to(torch.uint8)

    # Serialize tensor and get binary
    buffer = io.BytesIO()
    torch.save(tensor_1d, buffer)
    tensor_binary = buffer.getvalue()

    return tensor_binary


def chunk_binobj(tensor_compressed, id, kind, chunksize):
    # Convert chunksize from megabytes to bytes
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

def reset_counter_and_collection(main_collection):
    # Delete all documents in the main collection that have creeped in
    # between the renaming and now. This operation is within a
    # transaction
    main_collection.delete_many({})
    # Reset the counter to zero
    result = db[COLLECTIONc].update_one(
        {"_id": "uniqueFieldCounter"},  # Query part: the document to match
        {"$set": {"sequence_value": 0}},  # Update part: what to set if the document is matched/found
        upsert=True  # This ensures that if the document doesn't exist, it will be inserted
    )
    # Delete all documents in the main collection that have creeped in
    # between the renaming and now. This operation is within a
    # transaction
    main_collection.delete_many({})

"""
def reset_counter_and_collection_transaction():
    # Attempting logical atomicity
    with client.start_session() as session:
        with session.start_transaction():
            # Reset the counter to zero
            db[COLLECTIONc].update_one(
                {"_id": "uniqueFieldCounter"},
                {"$set": {"sequence_value": 0}},
                session=session,
            )

            # Delete all documents in the main collection that have creeped in
            # between the renaming and now. This operation is within a
            # transaction
            main_collection.delete_many({}, session=session)

            # Commit the transaction
            session.commit_transaction()
"""

# Function to watch the counter and perform actions when a threshold is reached
def watch_and_swap(TARGET_COUNTER_VALUE, generated, LOG_METRICS=False):
    if LOG_METRICS: swap_start = time.time()
    counter_doc = db[COLLECTIONc].find_one({"_id": "uniqueFieldCounter"})
    if LOG_METRICS: check_counter_end = time.time()
    #print("checked the counter ", counter_doc["sequence_value"], flush=True)
    if counter_doc["sequence_value"] >= TARGET_COUNTER_VALUE:
        # Actions to be taken when the threshold is reached
        # Renaming the collection and creating a new one
        time.sleep(2)
        generated += TARGET_COUNTER_VALUE
        print(f"Generated samples so far {generated}", flush=True)
        db[COLLECTIONw].rename(COLLECTIONt, dropTarget=True)
        # Now atomically reset the counter to 0 and delete whatever records
        # may have been written between the execution of the previous line
        # and the next

        if LOG_METRICS: reset_start = time.time()
        reset_counter_and_collection(db[COLLECTIONw])  # this is atomic
        if LOG_METRICS: reset_end = time.time()
        # index temp.bin collection on id
        if LOG_METRICS: index_start = time.time()
        db[COLLECTIONt].create_index([("id", ASCENDING)])
        if LOG_METRICS: index_end = time.time()
        # delete all records with id > (TARGET_COUNTER_VALUE - 1)
        if LOG_METRICS: clean_start = time.time()
        result = db[COLLECTIONt].delete_many(
            {"id": {"$gt": TARGET_COUNTER_VALUE - 1}}
        )
        if LOG_METRICS: clean_end = time.time()
        # Print the result of the deletion
        print(f"Documents deleted: {result.deleted_count}", flush=True)
        if LOG_METRICS: verify_start = time.time()
        assert_sequence(db[COLLECTIONt], TARGET_COUNTER_VALUE)
        db[COLLECTIONt].rename(COLLECTIONr, dropTarget=True)
        if LOG_METRICS: verify_end = time.time()
        if LOG_METRICS: swap_end = time.time()
    

        if LOG_METRICS: # For printing out timing stats, unclear if this is useful for paper metrics
            print(f"Swap operation took     {swap_end-swap_start} seconds")
            print(f"Checking counter took   {check_counter_end-swap_start} seconds")
            print(f"Resetting took          {reset_end-reset_start} seconds")
            print(f"Indexing took           {index_end-index_start} seconds")
            print(f"Cleaning operation took {clean_end-clean_start} seconds")
            print(f"Verifying took          {verify_end-verify_start} seconds")

        if LOG_METRICS:
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
            """
            Other useful metrics:
            - Resource Util: IO/CPU/Network on mongo node
            - Metrics per worker: For different hardware configs and software optimizations
            Everything else that can be derived:
            - Samples/second
            - Seconds/swap
            - Samples/second/worker
            - Data volume moved (there's a caveat of reads being hard to measure without
            putting load on worker)
            """

        
    return generated


def generate_and_insert(
    brain_generator, collection_bin, counter_collection, chunk_size
):
    img, lab = brain_generator.generate_brain()
    img = preprocess_image_min_max(img) * 255
    img = img.astype(np.uint8)
    lab = preprocess_label(lab)

    img_tensor = tensor2bin(torch.from_numpy(img))
    lab_tensor = tensor2bin(torch.from_numpy(lab))

    counter_doc = counter_collection.find_one_and_update(
        {"_id": "uniqueFieldCounter"},
        {"$inc": {"sequence_value": 1}},
        return_document=ReturnDocument.BEFORE,
    )
    index = counter_doc["sequence_value"]
    chunks = list(chunk_binobj(img_tensor, index, "data", chunk_size)) + list(
        chunk_binobj(lab_tensor, index, "label", chunk_size)
    )

    try:
        collection_bin.insert_many(chunks)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        print(f"I expect you are renaming the collection", flush=True)
        time.sleep(1)

    # with client.start_session() as session:
    #     with session.start_transaction():
    #         counter_doc = counter_collection.find_one_and_update(
    #             {"_id": "uniqueFieldCounter"},
    #             {"$inc": {"sequence_value": 1}},
    #             return_document=ReturnDocument.BEFORE,
    #             session=session,
    #         )
    #         index = counter_doc["sequence_value"]
    #         chunks = list(
    #             chunk_binobj(img_tensor, index, "T1", chunk_size)
    #         ) + list(chunk_binobj(lab_tensor, index, "label", chunk_size))

    #         try:
    #             collection_bin.insert_many(chunks, session=session)
    #             # Only commit the transaction if no exceptions are thrown
    #             session.commit_transaction()
    #         except Exception as e:
    #             # Abort the transaction on error
    #             session.abort_transaction()
    #             print(f"An error occurred: {e}")
    #             print(f"I expect you are renaming the collection")
    #             time.sleep(1)


if __name__ == "__main__":
    print("".join(["-"] * 50), flush=True)
    task_id = my_task_id()

    if is_first_job() or LOCAL_RUNNING:
        print(
            "I am the manager "
            + "\U0001F468\U0001F4BC"
            + ", watching the bottomline.",  # Also doing some work.",
            flush=True,
        )
        reset_counter_and_collection(db[COLLECTIONw])
        generated = 0
        while True:
            generated = watch_and_swap(TARGET_COUNTER_VALUE, generated, LOG_METRICS=LOG_METRICS)
            # generate_and_insert(
            #     brain_generator, db[COLLECTIONw], db[COLLECTIONc], CHUNKSIZE
            # )
    else:
        brain_generator = create_generator(task_id)
        print(
            "I am a worker "
            + "\U0001F41D"
            + ", generating and inserting data.",
            flush=True,
        )
        while True:
            generate_and_insert(
                brain_generator, db[COLLECTIONw], db[COLLECTIONc], CHUNKSIZE
            )
            # time.sleep(1)  # Adjust the sleep time as needed
