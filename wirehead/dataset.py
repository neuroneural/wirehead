import io
import os
import time
import yaml
import torch

from torch.utils.data import Dataset, get_worker_info
from pymongo import MongoClient
from pymongo.errors import OperationFailure


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
        :param sample: a pair of fields to be fetched as `input` and `label`, e.g. (`T1`, `label104`)
        :param id: the field to be used as an index. The `indices` are values of this field
        :param keeptrying: whether to keep retrying to fetch a record if the process failed or just report this and fail
        :returns: an object of MongoheadDataset class
        """
        self.id = id
        self.normalize = normalize
        self.transform = transform
        self.keeptrying = keeptrying    # retries if fetch fails
        self.fields = {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1}

        if config_path != "" and os.path.exists(config_path):
            self.load_from_yaml(config_path)

        else:
            self.collection = collection
            self.sample = sample

        self.indices = self.get_indeces()

    def load_from_yaml(self, config_path):
        """ Loads config options from config_path """
        print("Dataset: config loaded from " + config_path)
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        DBNAME = config.get("DBNAME")
        MONGOHOST = config.get("MONGOHOST")
        client = MongoClient("mongodb://" + MONGOHOST + ":27017")
        db = client[DBNAME]
        self.wait_for_data(db)
        read_collection = config.get("READ_COLLECTION")

        self.collection = db[read_collection]
        self.sample = tuple(config.get("SAMPLE"))

    def wait_for_data(self, db):
        status_collection = db["status"]
        latest_status = status_collection.find_one(sort=[("_id", -1)])
        while latest_status == None:
            latest_status = status_collection.find_one(sort=[("_id", -1)])
            print("Dataset: Database does not exist, hanging")
            time.sleep(5)

        swapped = latest_status.get("swapped")
        while swapped == False:
            latest_status = status_collection.find_one(sort=[("_id", -1)])
            swapped = latest_status.get("swapped")
            print("Dataset: Swap has not happened, hanging")
            time.sleep(5)

    def get_indeces(self):
        last_post = self.collection['bin'].find_one(sort=[(self.id, -1)])

        if last_post is None:
            print("Empty collection, exiting")
            exit()
        num_examples = int(last_post[self.id] + 1)
        return range(num_examples)

    def __len__(self):
        return len(self.indices)

    def make_serial(self, samples_for_id, kind):
        return b"".join([
            sample["chunk"] for sample in sorted(
                (sample for sample in samples_for_id if sample["kind"] == kind),
                key=lambda x: x["chunk_id"],
            )
        ])

    def retry_on_eof_error(retry_count, verbose=False):

        def decorator(func):

            def wrapper(self, batch, *args, **kwargs):
                myException = Exception    # Default Exception if not overwritten
                for attempt in range(retry_count):
                    try:
                        return func(self, batch, *args, **kwargs)
                    except (
                            EOFError,
                            OperationFailure,
                    ) as e:    # Specifically catching EOFError
                        if self.keeptrying:
                            if verbose:
                                print(
                                    f"EOFError caught. Retrying {attempt+1}/{retry_count}"
                                )
                            time.sleep(1)
                            myException = e
                            continue
                        else:
                            raise e
                raise myException("Failed after multiple retries.")

            return wrapper

        return decorator

    @retry_on_eof_error(retry_count=3, verbose=True)
    def __getitem__(self, batch):
        """ Fetch all samples for ids in the batch and where 'kind' is either
            data or label as specified by the sample parameter """
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
