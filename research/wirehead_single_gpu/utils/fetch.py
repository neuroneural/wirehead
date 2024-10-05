import torch
from itertools import islice
from torch.utils.data import DataLoader
from mindfultensors.gencoords import CoordsGenerator
from mindfultensors.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    DBBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)


def get_eval_dataloader():
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

# dspecifying the database location and collection name
    LABELFIELD = "label104"
    DATAFIELD = "T1"
    n_classes = 104

    MONGOHOST = "arctrdcn018.rs.gsu.edu"
    DBNAME = "MindfulTensors"
    COLLECTION = "HCPnew"  # Mindboggle and HCPnew are also in this new format
# index field and labels to retrieve
    INDEX_ID = "id"

    SAMPLES = 16  # subcubes per subject to sample
# percent of the data in a collection to use for validation
    validation_percent = 0.1

# specify dimension of the larger volume
    volume_shape = [256] * 3
# specify dimension of the subvolume
    subvolume_shape = [256] * 3
    coord_generator = CoordsGenerator(volume_shape, subvolume_shape)


    def unit_interval_normalize(img):
        """Unit interval preprocessing"""
        img = (img - img.min()) / (img.max() - img.min())
        return img


# wrapper functions
    def createclient(x):
        return create_client(
            x, dbname=DBNAME, colname=COLLECTION, mongohost=MONGOHOST
        )


    def mycollate_full(x):
        return mcollate(x)


    def mycollate(x):
        return collate_subcubes(x, coord_generator, samples=SAMPLES)


    def mytransform(x):
        return mtransform(x)


    client = MongoClient("mongodb://" + MONGOHOST + ":27017")
    db = client[DBNAME]
    posts = db[COLLECTION + ".meta"]
    col = db[COLLECTION]

# compute how many unique INDEX_ID values are present in the collection
# these are unique subjects
    num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

    tdataset = MongoDataset(
        range(int((1 - validation_percent) * num_examples)),
        mytransform,
        None,
        (DATAFIELD, LABELFIELD),
        normalize=lambda x: x,
        id=INDEX_ID,
    )

# We need a sampler that generates indices instead of trying to split the
# dataset into chunks
# use one subject at a time
    tsampler = DBBatchSampler(tdataset, batch_size=1)

# the standard pytorch class - ready to be used
    tdataloader = DataLoader(
        tdataset,
        sampler=tsampler,
        collate_fn=mycollate_full,
        # if you want the loader to place batch on GPU and at a fixed location
        # pin_memory=True,
        worker_init_fn=createclient,
        num_workers=1,  # currently does not work with <1
    )
    return tdataloader

def DK2synth(label, device):
    max_value = 103

    # Initialize the idx tensor with 1s
    idx = torch.ones(max_value + 1, dtype=torch.long).to(device)

    # Now set the other mappings
    idx[0] = 0
    idx[1:69] = 2
    idx[69:71] = 7
    idx[71:73] = 8
    idx[73:75] = 9
    idx[75:77] = 10
    idx[77:79] = 14
    idx[79:81] = 15
    idx[81:83] = 16
    idx[83:85] = 17
    idx[85:87] = 1
    idx[87] = 3
    idx[88] = 4
    idx[89] = 3
    idx[90] = 4
    idx[91] = 11
    idx[92] = 12
    idx[93] = 0
    idx[94] = 13
    idx[95:97] = 5
    idx[97:99] = 6

    return idx[label.long()]

def get_eval(dataloader, samples = 100):
    dataloader = get_eval_dataloader()
    samples = []
    for batch in islice(dataloader, n // dataloader.batch_size + 1):
        samples.extend(batch)
        if len(samples) >= n:
            return samples[:n]
    return samples


if __name__ == "__main__":
    dataloader = get_eval_dataloader()
    eval = iter(dataloader)
    img, lab = next(eval) 
    print(img.shape)
    print(img.dtype)
    print(lab.shape)
