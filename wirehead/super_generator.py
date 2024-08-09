""" Wirehead Manager + Generator Class """

import io
import os
import time
import yaml
import bson
import torch
from pymongo import MongoClient, ReturnDocument, ASCENDING
from pymongo.errors import OperationFailure, ConnectionFailure

class WireheadSuperGenerator:
    """
    Wirehead runtime class, which wraps around the generator
    and manager runtimes.
    """

    def __init__(self, generator, config_path, n_samples=int(1e9)):
        if config_path is None or os.path.exists(config_path) is False:
            print("No valid config specified, exiting")
            return
        self.load_from_yaml(config_path)
        self.generator = generator
        self.n_samples = n_samples
        self.check_and_reinitialize_database()
        self.delete_temp()

    def initialize_database(self):
        """Initialize the database with required collections"""
        # Initialize counters collection
        counters_collection = self.db[self.collectionc]
        counters_collection.insert_one({"_id": "uniqueFieldCounter", "sequence_value": 0})
        print(f"Initialized {self.collectionc} with sequence_value: 0")

        # Create write collection
        write_collection = self.db[self.collectionw]
        write_collection.create_index([("id", ASCENDING)], background=True)
        print(f"Created empty {self.collectionw} collection with index on 'id'")

        # Create status collection (empty)
        status_collection = self.db['status']
        print("Created empty status collection")

    def check_and_reinitialize_database(self):
        """Check the counters collection and reinitialize the database if the check fails"""
        counters_collection = self.db[self.collectionc]
        
        try:
            # Try to fetch the counter
            counter_doc = counters_collection.find_one({"_id": "uniqueFieldCounter"})
            
            if counter_doc is None:
                raise OperationFailure("Counter document not found")
            
        except OperationFailure:
            print("Counter fetch failed. Reinitializing database...")
            
            # Drop all collections in the database
            for collection_name in self.db.list_collection_names():
                self.db[collection_name].drop()
            
            # Reinitialize the database
            self.initialize_database()
            
            print("Database reinitialized successfully.")

    def load_from_yaml(self, config_path):
        """Loads manager configs from config_path"""
        with open(config_path, "r", encoding="utf-8") as file:
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


    def push_chunks(self, chunks):
        """Pushes chunkified tensors to mongodb, with error handling and ping"""
        collection_bin = self.db[self.collectionw]
        try:
            # Ping the write database with a small operation
            collection_bin.find_one({}, {"_id": 1})
            # If ping succeeds, insert the chunks
            collection_bin.insert_many(chunks)
        except (OperationFailure, ConnectionFailure) as exception:
            print(f"Generator: An error occurred: {exception}, write collection might not exist or there's a connection issue")
            time.sleep(1)

    def get_current_idx(self):
        """Get current index of sample in write collection"""
        dbc = self.db[self.collectionc]
        counter_doc = dbc.find_one_and_update(
            {"_id": "uniqueFieldCounter"},
            {"$inc": {"sequence_value": 1}},
            return_document=ReturnDocument.BEFORE,
        )
        if counter_doc == None:
            return 0
        return counter_doc["sequence_value"]

    def generate_and_insert(self):
        """Fetch from generator and inserts into mongodb"""
        # 0. Fetch data from generator
        data = next(self.generator)
        # 1. Turn the data into a list of serialized chunks with fake id
        chunks = self.chunkify(data, 0)
        # 2. Get the correct index for this current sample
        index = self.get_current_idx()
        if index < self.swap_cap:
            print(index, self.swap_cap)
            branded_chunks = [{**d, "id": index} for d in chunks]
            # 3. Push to mongodb + error handling
            self.push_chunks(branded_chunks)
        else:
            self.watch_and_swap(0)
            self.reset_counter_and_collection()

    def run_generator(self):
        """Initializes and runs a SynthSeg brain generator in a loop,
        preprocesses, then pushes to mongoDB"""
        print("Generator: Initialized")
        n_samples = self.n_samples
        for _ in range(n_samples):
            self.generate_and_insert()

    def verify_collection_integrity(self, collection):
        """
        Verifies collection contains contiguous elements with id 0..swap_cap
        """
        unique_ids_count = len(collection.distinct("id"))
        if unique_ids_count != self.swap_cap:
            print(
                f"Manager: Expected {self.swap_cap} unique ids, found {unique_ids_count}"
            )
            return False
        # can be factored to make faster (use numpy?)
        actual_ids_set = set(collection.distinct("id"))
        if self.expected_ids_set != actual_ids_set:
            print(
                "Manager: The ids aren't continuous from 0 to self.swap_cap - 1"
            )
            return False
        # If all checks pass
        return True

    def reset_counter_and_collection(self):
        """
        Delete all documents in the main collection that have creeped in
        between the renaming and now. This operation is within a transaction.
        """
        dbw = self.db[self.collectionw]
        dbc = self.db[self.collectionc]
        dbw.delete_many({})  # wipe the write collection
        # Reset the counter to zero
        _result = dbc.update_one(
            {"_id": "uniqueFieldCounter"},  # Query part: the document to match
            {
                "$set": {"sequence_value": 0}
            },  # Update part: what to set if the document is matched/found
            upsert=True,
        )
        dbw.delete_many({})
        dbw.create_index([("id", ASCENDING)], background=True)
    
    def delete_temp(self):
        dbt = self.db[self.collectiont]
        dbt.drop()

    def swap(self, generated):
        """
        Moves data from write collection to read collection
        Deletes old write collection
        Maintains data integrity in between
        """
        time.sleep(2)  # Buffer for incomplete ops
        generated += self.swap_cap
        try:
            self.db[self.collectionw].rename(self.collectiont, dropTarget=False)
            
            # Delete the write collection immediately after renaming
            self.db[self.collectionw].drop()
            
        except OperationFailure:
            print("Manager:Other manager swapping, swap skipped")
            return 


        result = self.db[self.collectiont].delete_many({"id": {"$gt": self.swap_cap - 1}})
        
        if self.verify_collection_integrity(self.db[self.collectiont]):
            self.db[self.collectiont].rename(self.collectionr, dropTarget=True)
            self.db["status"].insert_one({"swapped": True})

            print(f"Manager: Time: {time.time()} Generated samples so far {generated}")

            print(f"Manager: Documents deleted: {result.deleted_count}")
            print("====Manager: Swap success!===")
        else:
            print("Manager: Corrupted collection detected, swapskipped")
            self.db[self.collectionw].drop()
            self.check_and_reinitialize_database()
        
        
        # Create the write collection again
        self.db.create_collection(self.collectionw)
        self.db[self.collectionw].create_index([("id", ASCENDING)], background=True)
        self.delete_temp()

    def watch_and_swap(self, generated):
        """
        Watch the write collection and swap when full
        """
        counter_doc = self.db[self.collectionc].find_one(
            {"_id": "uniqueFieldCounter"}
        )
        idx = 0 if counter_doc == None else counter_doc["sequence_value"]
        if idx >= self.swap_cap:  # watch
            return self.swap(generated)  # swap
        return generated
