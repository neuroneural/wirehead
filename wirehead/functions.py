import io
import bson
from pymongo import MongoClient
import torch
import numpy as np
import pickle
import time
from wirehead.defaults import SWAP_THRESHOLD, CHUNKSIZE, NUMCHUNKS, DEFAULT_IMG, DEFAULT_LAB

###############################
### Functions for generator ###
###############################
def push_mongo(package_bytes,id,collection_bin,chunksize=CHUNKSIZE):
    """Pushes a chunkified serilized tuple containing two serialized (img: torch.Tensor, lab:torch.tensor)"""
    for chunk in chunk_binobj(package_bytes, id, chunksize):
        collection_bin.insert_one(chunk)

def chunk_binobj(tensor_compressed, id, chunksize):
    """Convert chunksize from megabytes to bytes"""
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
            "id": id,
            "chunk_id": i,
            "chunk": bson.Binary(chunk),
        }

def tensor2bin(tensor: torch.Tensor) -> bytes:
    """Serialize a torch tensor into uint8"""
    tensor = tensor.to(torch.uint8)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    tensor_binary = buffer.getvalue()
    return tensor_binary

def get_np_tensor_info(tensor: np.ndarray):
    """ Prints out information about a numpy ndarray"""
    min_value = np.min(tensor)
    max_value = np.max(tensor)
    shape = tensor.shape
    dtype = tensor.dtype
    print(f"Min Value : {min_value}")
    print(f"Max Value : {max_value}")
    print(f"Shape     : {shape}")
    print(f"Data Type : {dtype}")

def preprocess_image_quantile(img: np.ndarray, qmin=0.01, qmax=0.99)->np.ndarray:
    "Unit interval preprocessing for quantile normalization"
    qmin_value = np.quantile(img, qmin)
    qmax_value = np.quantile(img, qmax)
    img = (img - qmin_value) / (qmax_value - qmin_value)
    return img

def preprocess_image_min_max(img:np.ndarray)->np.ndarray:
    "Min max scaling preprocessing for the range 0..1"
    img = ((img - img.min()) / (img.max() - img.min()))
    return img

#############################
### Functions for manager ###
#############################
def swap_collections(db):
    global total_records
    start_time = time.time()
    # Create a new collection with a temporary name
    temp_collection_name = 'temp_collection'
    create_capped_collection(db, temp_collection_name, max_size=100000, max_documents=10000)
    package_ids = db['write'].distinct('id')    # Get the distinct package IDs in the 'write' collection
    for package_id in package_ids:              # Iterate over each package ID
        # Find all the chunks for the current package ID in the 'write' collection
        chunks = list(db['write'].find({'id': package_id}).sort([('chunk_id', 1)]))
        # Update the chunk IDs to maintain the insertion order
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            db[temp_collection_name].insert_one(chunk)
    db['read'].drop()                           # Drop the 'read' collection
    db['write'].rename('read')                  # Rename the 'write' collection to 'read'
    db[temp_collection_name].rename('write')    # Rename the temporary collection to 'write'
    total_records = 0                           # reset the count for the database
    print("Swap operation completed in", time.time() - start_time, "seconds")
    print(f"Total samples generated: {total_records}")

def create_capped_collection(db, collection_name, max_samples):
    # Calculate the maximum size based on the number of samples
    max_size = max_samples * ((256*256*256) * 2) + 1024*10# Packet size + a buffer
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name, capped=True, size=max_size)
    return db[collection_name]

def remove_invalid_chunks(collection, expected_chunks=NUMCHUNKS, DEBUG=False):
    # Get all distinct IDs in the collection
    distinct_ids = collection.distinct('id')
    # Iterate over each distinct ID
    for id_value in distinct_ids:
        # Count the number of chunks associated with the current ID
        chunk_count = collection.count_documents({'id': id_value})
        # Check if the chunk count matches the expected number of chunks
        if chunk_count != expected_chunks:
            # Remove all documents with the invalid ID
            collection.delete_many({'id': id_value})
            if DEBUG: print(f"Removed {chunk_count} chunks for ID: {id_value}")
    if DEBUG: print("Finished removing invalid chunks.")

def get_mongo_bytes(db):
    """Returns the size of mongo read and write halves in bytes"""
    raise NotImplementedError()

def get_mongo_write_size(db):
    """Returns the number of samples in the write half of mongo"""
    raise NotImplementedError()

#############################
### Functions for dataset ###
#############################
def safe_fetch( collection_bin, id_iterator, 
                nchunks=NUMCHUNKS, max_fetches = 10,
                fetches = 0, DEBUG=False) -> (torch.Tensor, torch.Tensor):
    """Safely returns a image label pair from a mongodb collection"""
    chunks = chunks = list(collection_bin.find({"id": next(id_iterator)}).sort("chunk_id"))
    while (len(chunks) != nchunks and fetches < max_fetches):
        chunks = safe_fetch(collection_bin, id_iterator)
        fetches += 1 
    if fetches >= max_fetches:
        return DEFAULT_IMG, DEFAULT_LAB
    data = b"".join(chunk["chunk"] for chunk in chunks)
    package = pickle.loads(data)
    img = bin2tensor(package[0])
    lab = bin2tensor(package[1])
    if DEBUG: print(f"{time.time()} {chunks[0]['id']}")
    return img, lab

def id_iterator(collection_bin, DEBUG = False) -> int:
    """Yields a valid id from the current collection, hopefully safely"""
    idx = 0
    id_list = collection_bin.distinct('id')
    while True:
        try:
            if DEBUG: print(f'Debug: {len(id_list)}')
            yield id_list[idx % SWAP_THRESHOLD]
            id_list = collection_bin.distinct('id')
            idx = (idx + 1) % SWAP_THRESHOLD
        except:
            if DEBUG: print(f'Debug: idx out of range')
            """Note to self: this can lead to unintended consequences 
            when plugged into the sample fetcher. There are no guarantees
            that the idx 0 will have all of its chunks in the database"""
            idx = 0
            yield id_list[idx]
   
def bin2tensor(binary_data):
    """Deserialize a binary buffer into a torch tensor"""
    buffer = io.BytesIO(binary_data)
    tensor = torch.load(buffer)
    return tensor


if __name__=="__main__":
    print("This file contains functions that are used by other components of wirehead")

