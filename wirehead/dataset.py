""" Wirehead dataset class for MongoDB """

import os
import sys
import time
import yaml
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ConnectionFailure

from mindfultensors.mongoloader import MongoheadDataset


class WireheadDataset(MongoheadDataset):
    """
    A dataset for fetching batches of records from a MongoDB.
    Adds wirehead's yaml config loading and swap-awareness on top of
    mindfultensors' dtype-dispatched, retry-resilient MongoDataset.
    """

    def __init__(self,
                 config_path="",
                 timeout=60,
                 collection=None,
                 sample=("data", "label"),
                 normalize=lambda x: x,
                 transforms=None,
                 id="id",
                 keeptrying=True):
        """Constructor
        :param config_path: path to wirehead config .yaml file
        :param collection: pymongo collection to be used
        :param sample: a tuple of kind names to be fetched, e.g. (`data`, `label`)
        :param normalize: a function applied to each tensor kind after transform
        :param transforms: optional dict mapping dtype strings to transform functions,
                            overriding mindfultensors' DEFAULT_TRANSFORMS
        :param id: the field to be used as an index. The `indices` are values of this field
        :param keeptrying: whether to attempt a refetch if first attempt fails
        :returns: an object of WireheadDataset class
        """
        self.id = id
        self.timeout = timeout

        if config_path != "" and os.path.exists(config_path):
            self.load_from_yaml(config_path)

        else:
            self.collection = collection
            self.sample = sample

        super().__init__(
            indices=self.get_indeces(),
            collection=self.collection["bin"],
            fetch=self.sample,
            transforms=transforms,
            normalize=normalize,
            id=id,
            keeptrying=keeptrying,
        )

    def load_from_yaml(self, config_path):
        """
        Loads config options from config_path
        """
        print("Dataset: config loaded from " + config_path)
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        dbname = config.get('DBNAME')
        mongohost = config.get('MONGOHOST')
        port = config.get('PORT') if config.get('PORT') is not None else 27017
        client = MongoClient("mongodb://" + mongohost + ":" + str(port))

        db = client[dbname]
        self.wait_for_data(db)
        read_collection = config.get("READ_COLLECTION")
        self.collection = db[read_collection]
        self.sample = tuple(config.get("SAMPLE"))

    def wait_for_data(self, db, timeout=600, check_interval=10):
        """
        Prevents data object from reading before data is ready.
        Raises:
            TimeoutError: If the wait time exceeds the timeout
        """
        status_collection = db["status"]
        start_time = time.time()

        def check_status():
            try:
                latest_status = status_collection.find_one(sort=[("_id", -1)])
                if latest_status is None:
                    print("Dataset: Database is empty, waiting for data...")
                    return False
                if not latest_status.get("swapped", False):
                    print("Dataset: Swap has not happened, waiting...")
                    return False
                return True
            except (ConnectionFailure, OperationFailure) as e:
                print(f"Dataset: Error accessing database: {e}")
                return False

        while time.time() - start_time < timeout:
            if check_status():
                print("Dataset: Data is ready")
                return
            time.sleep(check_interval)

        raise TimeoutError("Dataset: Waited too long for data to be ready")

    def get_indeces(self):
        """
        Retrieve the index array of samples in read collection
        """
        last_post = self.collection['bin'].find_one(sort=[(self.id, -1)])

        if last_post is None:
            print("Empty collection, exiting")
            sys.exit()
        num_examples = int(last_post[self.id] + 1)
        return range(num_examples)

