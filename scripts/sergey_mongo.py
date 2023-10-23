class MongoBatchDataset(Dataset):
    def __init__(self, indices, transform, collection, fields=None, id="id"):
        self.indices = indices
        self.transform = transform
        self.collection = collection
        self.fields = fields
        self.id = id
    def __len__(self):
        return (1e6)
    def __getitem__(self, batch):
        if self.fields is None:
            field_list = {}
        else:
            field_list = {_: 1 for _ in self.fields}
        samples = self.collection.find(
            {self.id: {"$in": [self.indices[_] for _ in batch]}}, field_list
        )
        return [self.transform(_) for _ in samples]

