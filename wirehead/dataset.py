""" Wirehead dataset class for MongoDB """

import io
import os
import sys
import time
import yaml
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from pymongo.errors import OperationFailure, OperationFailure


def unit_interval_normalize(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


def quantile_normalize(img, qmin=0.01, qmax=0.99):
    """Unit interval preprocessing"""
    img = (img - img.quantile(qmin)) / (img.quantile(qmax) - img.quantile(qmin))
    return img


def binary_to_tensor(tensor_binary):
    """ Converts a binary io buffer to a torch tensor """
    buffer = io.BytesIO(tensor_binary)
    tensor = torch.load(buffer)
    return tensor


class MongoheadDataset(Dataset):
    """
    A dataset for fetching batches of records from a MongoDB
    """

    def __init__(self,
                 config_path="",
                 timeout=60,
                 collection=None,
                 sample=("data", "label"),
                 transform=binary_to_tensor,
                 normalize=lambda x: x,
                 id="id",
                 keeptrying=True):
        """Constructor
        :param config_path: path to wirehead config .yaml file
        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param sample: a pair of fields to be fetched as `input` and `label`
                       , e.g. (`T1`, `label104`)
        :param id: the field to be used as an index. The `indices` are values of this field
        :param keeptrying: whether to attempt a refetch if first attempt fails
        :returns: an object of MongoheadDataset class
        """
        self.id = id
        self.normalize = normalize
        self.transform = transform
        self.keeptrying = keeptrying    # retries if fetch fails
        self.fields = {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1}
        self.timeout = timeout

        if config_path != "" and os.path.exists(config_path):
            self.load_from_yaml(config_path)

        else:
            self.collection = collection
            self.sample = sample

        self.indices = self.get_indeces()

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

    def __len__(self):
        return len(self.indices)

    def make_serial(self, samples_for_id, kind):
        """
        Converts collection chunks into a contiguous byte sequence
        """
        return b"".join([
            sample["chunk"] for sample in sorted(
                (sample for sample in samples_for_id if sample["kind"] == kind),
                key=lambda x: x["chunk_id"],
            )
        ])

    def retry_on_eof_error(retry_count, verbose=False):
        """
        Error handling for reads that happen mid swap
        """

        def decorator(func):

            def wrapper(self, batch, *args, **kwargs):
                myException = Exception    # Default Exception if not overwritten
                for attempt in range(retry_count):
                    try:
                        return func(self, batch, *args, **kwargs)
                    except (
                            EOFError,
                            OperationFailure,
                    ) as exception:    # Specifically catching EOFError
                        if self.keeptrying:
                            if verbose:
                                print(
                                    f"EOFError caught. Retrying {attempt+1}/{retry_count}"
                                )
                            time.sleep(1)
                            continue
                        else:
                            raise exception
                raise myException("Failed after multiple retries.")

            return wrapper

        return decorator

    @retry_on_eof_error(retry_count=3, verbose=True)
    def __getitem__(self, batch):
        """
        Fetch all samples for ids in the batch and where 'kind' is either
        data or label as specified by the sample parameter
        """
        samples = list(self.collection["bin"].find(
            {
                self.id: {
                    "$in": [self.indices[_] for _ in batch]
                },
                "kind": {
                    "$in": self.sample
                },
            },
            self.fields,
        ))
        results = {}
        for id in batch:
            # Separate samples for this id
            samples_for_id = [
                sample for sample in samples if sample[self.id] == self.indices[id]
            ]

            # Separate processing for each 'kind'
            data = self.make_serial(samples_for_id, self.sample[0])
            label = self.make_serial(samples_for_id, self.sample[1])

            # Add to results
            results[id] = {
                "input": self.normalize(self.transform(data).float()),
                "label": self.transform(label),
            }
        return results

class MongoTupleheadDataset(MongoheadDataset):
    """
    A dataset for fetching batches of records from a MongoDB
    Returns a tuple instead of a dict
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @MongoheadDataset.retry_on_eof_error(retry_count=3, verbose=True)
    def __getitem__(self, batch):
        """
        Fetch all samples for ids in the batch and return as tuples
        """
        samples = list(self.collection["bin"].find(
            {
                self.id: {
                    "$in": [self.indices[_] for _ in batch]
                },
                "kind": {
                    "$in": self.sample
                },
            },
            self.fields,
        ))
        results = []
        for id in batch:
            samples_for_id = [
                sample for sample in samples if sample[self.id] == self.indices[id]
            ]

            data = self.make_serial(samples_for_id, self.sample[0])
            label = self.make_serial(samples_for_id, self.sample[1])

            data = self.normalize(self.transform(data).float())
            label = self.transform(label)

            results.append((data, label))
        return results
