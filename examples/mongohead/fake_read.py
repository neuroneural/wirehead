import torch
from tqdm import tqdm
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

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
LABELFIELD = "label"
DATAFIELD = "data"
MONGOHOST = "arctrdcn018.rs.gsu.edu"
DBNAME = "wirehead_test"
COLLECTION = "read"
INDEX_ID = "id"


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


def mytransform(x):
    return mtransform(x)


client = MongoClient("mongodb://" + MONGOHOST + ":27017")
db = client[DBNAME]
posts = db[COLLECTION]["bin"]


def get_max_id(collection):
    pipeline = [
        {
            "$group": {
                "_id": None,  # Grouping without a specific field to aggregate on the entire collection
                "unique_ids": {
                    "$addToSet": "$id"
                },  # Use $addToSet to get unique ids
                "max_id": {"$max": "$id"},  # Get the maximum id
            }
        }
    ]
    result = collection.aggregate(pipeline)
    for doc in result:
        max_id = doc["max_id"]
    return max_id


num_examples = get_max_id(posts) + 1
tdataset = MongoDataset(
    range(num_examples),
    mtransform,
    None,
    (DATAFIELD, LABELFIELD),
    normalize=unit_interval_normalize,
    id=INDEX_ID,
    persist_on_EOF=True,
)
tsampler = DBBatchSampler(tdataset, batch_size=1)
tdataloader = DataLoader(
    tdataset,
    sampler=tsampler,
    collate_fn=mycollate_full,
    # if you want the loader to place batch on GPU and at a fixed location
    # pin_memory=True,
    worker_init_fn=createclient,
    num_workers=1,  # currently does not work with <1
)

count = 0
while(True):
    for batch in tqdm(tdataloader):
        count += 1
