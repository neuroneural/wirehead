import time
import bson
import torch
import io
from pymongo import ReturnDocument

class Runtime():
    """ Wirehead runtime class, which wraps around the generator
        and manager runtimes."""
    def __init__(self,
                 db, 
                 generator, 
                 cap=1000, 
                 wcount=1, 
                 log_metrics=False):
        self.db                 = db
        self.generator          = generator 
        self.swap_cap           = cap
        self.CHUNKSIZE          = 10
        self.LOG_METRICS        = log_metrics
        self.EXPERIMENT_KIND    = ''
        self.EXPERIMENT_NAME    = ''
        self.WORKER_COUNT       = wcount
        self.LOCAL_RUNNING      = True
        self.COLLECTIONw        = "write.bin"
        self.COLLECTIONr        = "read.bin"
        self.COLLECTIONt        = "temp.bin"
        self.COLLECTIONc        = "counters"
        self.COLLECTIONm        = 'metrics'
    
    # Manager Ops
    def run_manager(self):
        """ Initializes the database manager, 
            swaps the mongo collections whenever 
            TARGET_COUNTER_VALUE is hit. """
        print(
            "".join(["-"] * 50)
            + "\nI am the manager "
            + "\U0001F468\U0001F4BC"
            + ", watching the bottomline.", 
            flush=True,)
        TARGET_COUNTER_VALUE = self.swap_cap
        dbw = self.db[self.COLLECTIONw]
        dbc = self.db[self.COLLECTIONc]
        LOG_METRICS = self.LOG_METRICS
        reset_counter_and_collection(db[COLLECTIONw], db[COLLECTIONc])
        generated = 0
        while True:
            generated = watch_and_swap(
                TARGET_COUNTER_VALUE, 
                generated, 
                LOG_METRICS=LOG_METRICS)

    def assert_sequence(self, collection):
        """Verify collection integrity"""
        unique_ids_count = len(collection.distinct("id"))
        assert (
            unique_ids_count == self.swap_cap
        ), f"Expected {self.swap_cap} unique ids, found {unique_ids_count}"
        expected_ids_set = set(range(self.swap_cap))
        actual_ids_set = set(collection.distinct("id"))
        assert (
            expected_ids_set == actual_ids_set
        ), "The ids aren't continuous from 0 to self.swap_cap - 1"

    def reset_counter_and_collection(write_collection, counter_collection):
        """ Delete all documents in the main collection that have creeped in
            between the renaming and now. This operation is within a
            transaction """
        write_collection.delete_many({})
        # Reset the counter to zero
        result = counter_collection.update_one(
            {"_id": "uniqueFieldCounter"},  # Query part: the document to match
            {
                "$set": {"sequence_value": 0}
            },  # Update part: what to set if the document is matched/found
            upsert=True,
        )
        # Delete all documents in the main collection that have creeped in
        # between the renaming and now. This operation is within a
        # transaction
        write_collection.delete_many({})
        write_collection.create_index([("id", ASCENDING)], background=True)

    def log_metrics(self, generated):
        """ Inserts run metrics into COLLECTIONm """
        metrics_collection  = self.db[COLLECTIONm]
        # Create a BSON document with the metrics
        metrics_doc = {
            "experiment_name": self.EXPERIMENT_NAME,
            "kind": self.EXPERIMENT_KIND,
            "timestamp": time.time(),
            "total_samples": generated,
            "worker_count": self.WORKER_COUNT
        }
        # Insert the metrics document into the metrics collection
        print('Manager: Swap metrics logged')
        metrics_collection.insert_one(metrics_doc)

    def watch_and_swap(self, generated):
        """ Watch the write collection and swap when full"""
        def swap(self, generated):
            """ Actions to be taken when the threshold is reached
                Renaming the collection and creating a new one """
            time.sleep(2) # Buffer for incomplete ops
            generated += self.swap_cap
            print(f"Generated samples so far {generated}", flush=True)
            self.db[self.COLLECTIONw].rename(self.COLLECTIONt, dropTarget=True)
            # Now atomically reset the counter to 0 and delete whatever records
            # may have been written between the execution of the previous line
            # and the next
            reset_counter_and_collection(
                self.db[self.COLLECTIONw], self.db[self.COLLECTIONc]
            )  # this is atomic
            result = self.db[self.COLLECTIONt].delete_many(
                {"id": {"$gt": self.swap_cap - 1}}
            )
            # Print the result of the deletion
            print(f"Documents deleted: {result.deleted_count}", flush=True)
            self.assert_sequence(self.db[self.COLLECTIONt], self.swap_cap)
            self.db[COLLECTIONt].rename(self.COLLECTIONr, dropTarget=True)

            if self.LOG_METRICS: log_metrics(generated)
            return generated

        counter_doc = self.db[self.COLLECTIONc].find_one(
            {"_id": "uniqueFieldCounter"})
        if counter_doc["sequence_value"] >= self.swap_cap:  # watch
            return swap(generated)                          # swap
        return generated

    # Generator Ops
    def chunkify(self, data, index, chunk_size):
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

        def tensor2bin(tensor):
            """Seralize a torch tensor into an IO buffer"""
            tensor_1d = tensor.to(torch.uint8)
            buffer = io.BytesIO()
            torch.save(tensor_1d, buffer)
            tensor_binary = buffer.getvalue()
            return tensor_binary

        chunks = []
        binobj, kinds = data
        for i, kind in enumerate(kinds):
            chunks += list(
                chunk_binobj(
                    tensor2bin(torch.from_numpy(binobj[i])),
                    index,
                    kind,
                    chunk_size)) 
        return chunks



    def push_chunks(self, collection_bin, chunks):
        """ Pushes chunkified tensors to mongodb, with error handling"""
        try:
            collection_bin.insert_many(chunks)
        except Exception as e:
            print(f"An error occurred: {e}", flush=True)
            print(f"I expect you are renaming the collection", flush=True)
            time.sleep(1)

    def get_current_idx(self, counter_collection):
        counter_doc = counter_collection.find_one_and_update(
            {"_id": "uniqueFieldCounter"},
            {"$inc": {"sequence_value": 1}},
            return_document=ReturnDocument.BEFORE,
        )
        return counter_doc["sequence_value"]


    def generate_and_insert(self,
                            collection_bin,
                            counter_collection,
                            chunk_size):
        """ Fetch from generator and inserts into mongodb """
        # 0. Fetch data from generator
        data = next(self.generator)
        # 1. Get the correct index for this current sample
        index = self.get_current_idx(counter_collection)
        # 2. Turn the data into a list of serialized chunks  
        chunks = self.chunkify(data, index, chunk_size)
        # 3. Push to mongodb + error handling
        self.push_chunks(collection_bin, chunks)

    def run_generator(self):
        """ Initializes and runs a SynthSeg brain generator in a loop,
            preprocesses, then pushes to mongoDB"""
        print(
            "".join(["-"] * 50)
            + "\nI am a worker "
            + "\U0001F41D"
            + ", generating and inserting data.",
            flush=True,
        )
        while True:
            print("ding") #TODO: debug gen loop
            dbw = self.db[self.COLLECTIONw]
            dbc = self.db[self.COLLECTIONc]
            chunksize = self.CHUNKSIZE
            self.generate_and_insert(dbw, dbc, chunksize)


if __name__ == "__main__":
    from pymongo import MongoClient
    import numpy as np
    import os
    import sys

    # Mongo config
    DBNAME              = "wirehead_mike"
    MONGOHOST           = "arctrdcn018.rs.gsu.edu"
    client              = MongoClient("mongodb://" + MONGOHOST + ":27017")
    db                  = client[DBNAME]

    # Synthseg config
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
    PATH_TO_DATA        = ("/data/users1/mdoan4/wirehead/dependencies/synthseg/data/training_label_maps/")
    DATA_FILES          = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
    PATH_TO_SYNTHSEG    = '/data/users1/mdoan4/wirehead/dependencies/synthseg'


    # Create a generator function that yields desired samples
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

    # Extras
    def my_task_id() -> int:
        """ Returns slurm task id """
        task_id = os.getenv(
            "SLURM_ARRAY_TASK_ID", "0"
        )  # Default to '0' if not running under Slurm
        return int(task_id)

    # Plug into wirehead 
    brain_generator     = create_generator(my_task_id())
    wirehead_runtime    = Runtime(
        db = db,                    # Specify mongohost
        generator = brain_generator,# Specify generator 
    )

    wirehead_runtime.run_generator()

    print(0)
