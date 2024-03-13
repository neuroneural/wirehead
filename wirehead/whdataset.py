import torch
from torch.utils.data import Dataset
import pymongo
import uuid

class whdataset(Dataset):
    def __init__(self, mongo_uri, database_name, collection_name, max_size, max_documents, num_chunks):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection_name = collection_name
        self.collection = self.create_capped_collection(max_size, max_documents)
        self.num_chunks = num_chunks

    def create_capped_collection(self, max_size, max_documents):
        if self.collection_name not in self.db.list_collection_names():
            self.db.create_collection(self.collection_name, capped=True, size=max_size, max=max_documents)
        return self.db[self.collection_name]

    def __len__(self):
        return self.collection.count_documents({}) // self.num_chunks

    def __getitem__(self, idx):
        # Retrieve the oldest package from the capped collection
        package_id = self.collection.find_one({}, sort=[('_id', pymongo.ASCENDING)])['package_id']

        # Retrieve all chunks for the package
        chunks = list(self.collection.find({'package_id': package_id}).sort('chunk_id'))

        if len(chunks) == self.num_chunks:
            # Reassemble the package from chunks
            reassembled_data = self.reassemble_package(chunks)

            # Process the reassembled package
            processed_package = self.process_package(reassembled_data)

            # Generate a new unique identifier for the processed package
            unique_id = str(uuid.uuid4())

            # Insert the processed package back into the capped collection
            self.collection.insert_one({
                'unique_id': unique_id,
                'data': processed_package
            })

            # Remove the original chunks from the collection
            self.collection.delete_many({'package_id': package_id})

            return processed_package
        else:
            raise IndexError("Incomplete package found in the dataset.")

    def reassemble_package(self, chunks):
        # Reassemble the package from chunks
        reassembled_data = b''
        for chunk in chunks:
            reassembled_data += chunk['data']
        return reassembled_data

    def process_package(self, package_data):
        # Perform the actual package processing
        processed_package = torch.tensor(package_data)
        return processed_package

    def add_chunk(self, chunk, package_id, chunk_id):
        # Insert a new chunk into the capped collection
        self.collection.insert_one({
            'package_id': package_id,
            'chunk_id': chunk_id,
            'data': chunk
        })

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
        normalize,#=unit_interval_normalize,
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