""" Wirehead Generator Class """

import io
import os
import time
import yaml
import bson
import torch
from pymongo import MongoClient, ReturnDocument


class WireheadGenerator():
    """
    Wirehead runtime class, which wraps around the generator
    and manager runtimes.
    """

    def __init__(self, generator, config_path, n_samples = 1000):
        if config_path is None or os.path.exists(config_path) is False:
            print("No valid config specified, exiting")
            return
        self.load_from_yaml(config_path)
        self.generator = generator
        self.n_samples = n_samples

    def load_from_yaml(self, config_path):
        """ Loads manager configs from config_path """
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        dbname = config.get('DBNAME')
        mongohost = config.get('MONGOHOST')
        port = config.get('PORT') if config.get('PORT') is not None else 27017
        client = MongoClient("mongodb://" + mongohost + ":" + str(port))

        self.db = client[dbname]
        self.swap_cap = config.get('SWAP_CAP')
        self.sample = tuple(config.get("SAMPLE"))
        self.chunksize = config.get("CHUNKSIZE")
        self.collectionw = config.get("WRITE_COLLECTION") + ".bin"
        self.collectionc = config.get("COUNTER_COLLECTION")

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
                chunk_binobj(tensor2bin(torch.from_numpy(binobj[i])), index, kind,
                             self.chunksize))
        return chunks

    def push_chunks(self, chunks):
        """ Pushes chunkified tensors to mongodb, with error handling"""
        collection_bin = self.db[self.collectionw]
        try:
            collection_bin.insert_many(chunks)
        except Exception as exception:
            print(f"Generator: An error occurred: {exception}, are you swapping?")
            time.sleep(1)

    def get_current_idx(self):
        """ Get current index of sample in write collection """
        dbc = self.db[self.collectionc]
        counter_doc = dbc.find_one_and_update(
            {"_id": "uniqueFieldCounter"},
            {"$inc": {
                "sequence_value": 1
            }},
            return_document=ReturnDocument.BEFORE,
        )
        return counter_doc["sequence_value"]

    def generate_and_insert(self):
        """ Fetch from generator and inserts into mongodb """
        # 0. Fetch data from generator
        data = next(self.generator)
        # 1. Get the correct index for this current sample
        index = self.get_current_idx()
        # 2. Turn the data into a list of serialized chunks
        chunks = self.chunkify(data, index)
        # 3. Push to mongodb + error handling
        if index < self.swap_cap:
            self.push_chunks(chunks)

    def run_generator(self):
        """ Initializes and runs a SynthSeg brain generator in a loop,
            preprocesses, then pushes to mongoDB"""
        print("Generator: Initialized")
        n_samples = self.n_samples
        for _ in range(n_samples):
            self.generate_and_insert()
