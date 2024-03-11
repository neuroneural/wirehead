from pymongo import MongoClient
import torch
from torch.utils.data import Dataset
import io

# Assuming you have a MongoDB client and database set up
client = MongoClient('mongodb://10.245.12.58:27017/')
db = client["wirehead_test"]
collection_bin = db["read"]

class MongoDataset(Dataset):
    """
    A dataset for fetching batches of records from a MongoDB
    """

    def __init__(
        self,
        indices,
        transform,
        collection,
        sample,
        normalize,
        id="id",
    ):
        """Constructor

        :param indices: a set of indices to be extracted from the collection
        :param transform: a function to be applied to each extracted record
        :param collection: pymongo collection to be used
        :param sample: a pair of fields to be fetched as `input` and `label`, e.g. (`T1`, `label104`)
        :param id: the field to be used as an index. The `indices` are values of this field
        :returns: an object of MongoDataset class

        """

        self.indices = indices
        self.transform = transform
        self.collection = collection
        # self.fields = {_: 1 for _ in self.fields} if fields is not None else {}
        self.fields = {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1}
        self.sample = sample
        self.normalize = normalize
        self.id = id

    def __len__(self):
        return len(self.indices)

    def make_serial(self, samples_for_id, kind):
        return b"".join(
            [
                sample["chunk"]
                for sample in sorted(
                    (
                        sample
                        for sample in samples_for_id
                        if sample["kind"] == kind
                    ),
                    key=lambda x: x["chunk_id"],
                )
            ]
        )

    def __getitem__(self, batch):
        print(batch)
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or labela s specified by the sample parameter
        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": [self.indices[_] for _ in batch]},
                    "kind": {"$in": self.sample},
                },
                self.fields,
            )
        )
        print(len(samples))

        results = {}
        for id in batch:
            # Separate samples for this id
            samples_for_id = [
                sample
                for sample in samples
                if sample[self.id] == self.indices[id]
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

# Define the necessary functions and parameters
def mtransform(tensor_binary):
    buffer = io.BytesIO(tensor_binary)
    tensor = torch.load(buffer)
    return tensor

def unit_interval_normalize(tensor):
    return tensor / 255.0
# Create an instance of the MongoDataset class
indices = [0, 1, 2, 3, 4]  # Example indices to fetch
sample = ("data", "label")  # Specify the fields to fetch as input and label

dataset = MongoDataset(
    indices=indices,
    transform=mtransform,
    collection=collection_bin,
    sample=sample,
    normalize=unit_interval_normalize,
    id="id"
)

# Fetch a batch of samples from the dataset
batch_indices = [0, 2, 4]  # Example batch indices
batch_samples = dataset[batch_indices]
