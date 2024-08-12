""" Wirehead Generator Class """

import io
import os
import time
from typing import Any, Generator, Optional, Tuple

import yaml
import bson

import torch
from pymongo import MongoClient, ReturnDocument, ASCENDING
from pymongo.errors import OperationFailure, ConnectionFailure

class WireheadGenerator:
    """
    Wirehead generator class, which manages writes to mongodb.

    generator   : generator function, yields tuples of data
    config path : path to wirehead config file. default = "config.yaml"
    n_samples   : number of samples to generate. default = 1 billion
    """

    def __init__(self,
                 generator: Generator[Tuple[Any, ...], None, None],
                 config_path: str = "config.yaml",
                 n_samples: int = int(1e9)):
        if config_path is None or os.path.exists(config_path) is False:
            print("No valid config specified, exiting")
            return
        self.generator = generator
        self.n_samples = n_samples
        self.config_path = config_path
        self.load_from_yaml()
        self.reinitialize_database()

    def load_from_yaml(self):
        """
        Loads configs from config_path
        """
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        dbname = config.get("DBNAME")
        mongohost = config.get("MONGOHOST")
        port = config.get("PORT") if config.get("PORT") is not None else 27017
        client = MongoClient("mongodb://" + mongohost + ":" + str(port))

        self.db = client[dbname]
        self.swap_cap = config.get("SWAP_CAP")
        self.sample = tuple(config.get("SAMPLE"))
        self.chunksize = config.get("CHUNKSIZE")
        self.collectionw = config.get("WRITE_COLLECTION") + ".bin"
        self.collectionr = config.get("READ_COLLECTION") + ".bin"
        self.collectiont = config.get("TEMP_COLLECTION") + ".bin"
        self.collectionc = config.get("COUNTER_COLLECTION")
        self.expected_ids_set = set(range(self.swap_cap))


    def ping(self, collection_name: str) -> bool:
        """ Returns true if collection both exists and is reachable """
        try:
            if collection_name not in self.db.list_collection_names():
                return False
            self.db[collection_name].find_one({}, {'_id': 1})
            return True
        except:
            return False


    def reset_counter(self):
        counters_collection = self.db[self.collectionc]
        counters_collection.update_one(
            {"_id": "started"},
            {"$set": {"sequence_value": 0}},
            upsert=True
        ) # update start index
        counters_collection.update_one(
            {"_id": "completed"},
            {"$set": {"sequence_value": 0}},
            upsert=True
        ) # update comnpleted index

    def reset_write(self):
        if self.ping(self.collectionw):
            self.db.create_collection(self.collectionw)
            self.db[self.collectionw].create_index([("id", ASCENDING)], background=True)

    def reset_counter_and_write(self):
        self.reset_counter()
        self.reset_write()

    def reinitialize_database(self):
        """
        Check the counters collection and reinitialize the database if the check fails
        """
        # Drop all collections except the write collection
        for collection_name in [self.collectionw, self.collectiont, self.collectionc]:
            self.db[collection_name].drop()
        # Initialize counters collection
        self.reset_counter()
        # Create write collection
        try:
            write_collection = self.db[self.collectionw]
            write_collection.create_index([("id", ASCENDING)], background=True)
            # Create status collection (empty)
            _status_collection = self.db['status']
            print("Generator: Database reinitialized successfully.")
        except Exception as e:
            print(f"An error occurred while creating the index: {str(e)}")



    def temp_is_valid(self):
        """
        Verifies temp collection contains contiguous elements with id 0..swap_cap
        """
        temp_ids = self.db[self.collectiont].distinct("id")
        """ Checks if there are enough elements """
        unique_ids_count = len(temp_ids)
        if unique_ids_count != self.swap_cap:
            print(
                f"Generator: Expected {self.swap_cap} ids, found {unique_ids_count}"
            )
            return False
        """ Checks for contiguous ids """
        actual_ids_set = set(temp_ids)
        if self.expected_ids_set != actual_ids_set:
            print(
                "Generator: Ids aren't continuous from 0 to self.swap_cap - 1"
            )
            return False
        # If all checks pass
        return True

    
    def swap(self):
        """
        Moves data from write collection to read collection
        Deletes old write collection
        Maintains data integrity in between
        """
        try:
            self.db[self.collectionw].rename(self.collectiont, dropTarget=False)
            """
            Implicit mutex # 1.a lock create
            := Deletes the write collection
            := The nonexistence of the write collection can act as a lock
            := which prevents any writes or increments from happening.
            """
            self.db[self.collectionw].drop()

            result = self.db[self.collectiont].delete_many({"id": {"$gt": self.swap_cap - 1}})
            if self.temp_is_valid():
                self.db[self.collectiont].rename(self.collectionr, dropTarget=True)
                self.db["status"].insert_one({"swapped": True})
                """ Implicit mutex 1.b release """
                self.reset_write()

                print(f"Generator: Time: {time.time()}")
                print(f"Generator: Documents deleted: {result.deleted_count}")
                print("====Generator: Swap success!===============")

            dbt = self.db[self.collectiont]
            dbt.drop()
            
        except OperationFailure:
            print("Generator: Other manager swapping, swap skipped")
            return 


    def attempt_swap(self):
        """
        Watch the write collection and swap when full, with atomic mutex
        """
        counter_doc = self.db[self.collectionc].find_one({"_id": "completed"})
        idx = 0 if counter_doc is None else counter_doc["sequence_value"]
        # Don't attempt a swap if less than swap_cap
        if idx < self.swap_cap:
            return
        # Attempt the swap
        try:
            lock_doc = self.db[self.collectionc].find_one_and_update(
                {"_id": "swap_lock", "locked": False},
                {"$set": {"locked": True, "timestamp": time.time()}},
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
        except:
            time.sleep(0.1)
            print("Swap is locked, another instance is performing the swap operation.")
            return
        
        if lock_doc and lock_doc["locked"]:
            try:
                self.swap()                     # swap
                self.reset_counter_and_write()  # cleanup

            finally:
                # Release the lock
                self.db[self.collectionc].update_one(
                    {"_id": "swap_lock"},
                    {"$set": {"locked": False}}
                )
        else:
            print("Failed to acquire lock, another instance is performing the swap operation.")


    def get_idx(self, field="started", inc: int = 0): # other field is "completed"
        """Get current index of sample in write collection"""
        dbc = self.db[self.collectionc]
        counter_doc = dbc.find_one_and_update(
            {"_id": field},
            {"$inc": {"sequence_value": inc}},
            return_document=ReturnDocument.BEFORE,
        )
        if counter_doc is None:
            return 0
        return counter_doc["sequence_value"]


    def chunkify(self, data, index):
        """
        Converts a tuple of tensors and their labels into
        a list of chunks of serialized objects for mongodb
        """

        def chunk_binobj(tensor_compressed, idx, kind, chunksize):
            """
            Convert chunksize from megabytes to bytes
            """
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
                    "id": idx,
                    "chunk_id": i,
                    "kind": kind,
                    "chunk": bson.Binary(chunk),
                }

        def tensor2bin(tensor):
            """
            Seralize a torch tensor into an IO buffer
            """
            tensor_1d = tensor.to(torch.uint8)
            buffer = io.BytesIO()
            torch.save(tensor_1d, buffer)
            tensor_binary = buffer.getvalue()
            return tensor_binary

        chunks = []
        binobj = data
        kinds = self.sample
        for i, kind in enumerate(kinds):
            chunks += list(
                chunk_binobj(
                    tensor2bin(torch.from_numpy(binobj[i])),
                    index,
                    kind,
                    self.chunksize,
                )
            )
        return chunks


    def push(self, chunks):
        """
        Pushes chunkified tensors to mongodb, with error handling and ping
        """
        collection_bin = self.db[self.collectionw]
        try:
            """
            Implicit mutex # 2 explanation 
            :=  collection_bin.find_one({}, {"_id": 1}) will raise an error
                    if no write collection is found 
            :=  this lets us use the write collection itself as a mutex
            """
            # Ping the write database with a small operation
            collection_bin.find_one({}, {"_id": 1})
            # If ping succeeds, insert the chunks
            collection_bin.insert_many(chunks)
            # If push completes, increment completed counter
            _completed = self.get_idx(field="completed", inc=1)
        except (OperationFailure, ConnectionFailure) as exception:
            print(f"Generator: An error occurred: {exception}")
            time.sleep(1)


    def attempt_push(self):
        """Fetch from generator and inserts into mongodb"""
        # 0. Fetch data from generator
        data = next(self.generator)
        # 1. Turn the data into a list of serialized chunks with fake id
        chunks = self.chunkify(data, 0)
        # 2. Get the correct index for this current sample and increment index.
        index = self.get_idx(field="started", inc = 1) # this is atomic
        branded_chunks = [{**d, "id": index} for d in chunks]
        # 3. Push to mongodb + error handling
        if index < self.swap_cap:
            print(f"Pushing index: {index}, with cap: {self.swap_cap}")
            self.push(branded_chunks)
        self.attempt_swap()
        if index > self.swap_cap * 2:
            self.reinitialize_database()


    def run(self):
        """
        Attempts to insert (or) swap in a loop
        """
        print("Generator: Initialized")
        for _ in range(self.n_samples):
            self.attempt_push()
