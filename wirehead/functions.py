import io
import bson
from pymongo import MongoClient, UpdateMany
import torch
import numpy as np
import pickle
import time
import math
from multiprocessing import Pool
from wirehead.defaults import SWAP_THRESHOLD, CHUNKSIZE, NUMCHUNKS, DEFAULT_IMG, DEFAULT_LAB

###############################
### Functions for generator ###
###############################
def push_mongo(package, id, collection_bin, chunksize=CHUNKSIZE):
    """Pushes a chunkified serialized tuple containing two serialized (img: torch.Tensor, lab:torch.tensor)"""
    data_tensor, label_tensor = package
    data_bytes = tensor2bin(data_tensor)
    label_bytes = tensor2bin(label_tensor)

    for chunk in chunk_binobj(data_bytes, id, "data", chunksize):
        collection_bin.insert_one(chunk)

    for chunk in chunk_binobj(label_bytes, id, "label", chunksize):
        collection_bin.insert_one(chunk)

def gen_id_iterator(id_range):
    id_start, id_end = id_range
    current_id = id_start
    while True:
        yield current_id
        current_id += 1
        if current_id > id_end:
            current_id = id_start

def chunk_binobj(tensor_compressed, id, kind, chunksize):
    # Convert chunksize from megabytes to bytes
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
            "kind": kind,
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
def swap_db(db, DEBUG=False):
    if DEBUG: start_time = time.time()

    db['write']['bin'].rename('temp', dropTarget=True)                              # Create a temp record for processing
    temp = db['temp']
    create_capped_collection(db, 'write.bin', SWAP_THRESHOLD)   # Create a new write collection 
    drop_incomplete_samples(temp, ('data', 'label'))        # Drop incomplete packages from record 
    map_contiguous_ids3(temp)                                # Convert IDs into contiguous mapping
    db['temp'].rename('read.bin', dropTarget = True)            # Change the temp into the read collection

    if DEBUG: print("Swap operation completed in", time.time() - start_time, "seconds")

def monitor_insertions(db, DEBUG=False):
    global total_records
    total_records = 0
    write_collection = db['write']
    last_inserted_id = None

    while True:
        if last_inserted_id is None:
            query = {}
        else:
            query = {'_id': {'$gt': last_inserted_id}}

        new_records = list(write_collection.find(query))
        if new_records:
            last_inserted_id = new_records[-1]['_id']
            total_records += len(new_records)

            if total_records >= SWAP_THRESHOLD:
                swap_db(db, DEBUG=True)

        if DEBUG: print(f'Total records: {total_records}')
        time.sleep(1)  # Adjust the polling interval as needed

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

def drop_incomplete_samples(collection_bin, sample_kinds, expected_chunks=NUMCHUNKS):
    """Drops samples from the MongoDB collection that have incomplete chunks"""
    # Get all distinct sample IDs in the collection
    sample_ids = collection_bin.distinct("id")
    
    # Iterate over each sample ID
    for sample_id in sample_ids:
        # Count the number of data chunks for the current sample ID
        data_chunk_count = collection_bin.count_documents(
            {
                "id": sample_id,
                "kind": sample_kinds[0]
            }
        )
        # Count the number of label chunks for the current sample ID
        label_chunk_count = collection_bin.count_documents(
            {
                "id": sample_id,
                "kind": sample_kinds[1]
            }
        )
        # Check if the chunk counts match the expected number of chunks
        if data_chunk_count != expected_chunks or label_chunk_count != expected_chunks:
            # Drop all chunks associated with the incomplete sample ID
            collection_bin.delete_many(
                {
                    "id": sample_id
                }
            )
            print(f"Dropped sample ID: {sample_id}")
    print("Finished dropping incomplete samples.")


def map_contiguous_ids(collection_bin, DEBUG=False):
    if DEBUG: start = time.time()
    # Get all distinct sample IDs in the collection
    distinct_ids = collection_bin.distinct("id")
    
    # Sort the distinct IDs
    sorted_ids = sorted(distinct_ids)
    
    # Create a dictionary to map original sample IDs to contiguous IDs
    id_map = {id: contiguous_id for contiguous_id, id in enumerate(sorted_ids)}
    
    # Update each document in the collection with the contiguous ID
    for original_id, contiguous_id in id_map.items():
        collection_bin.update_many(
            {"id": original_id},
            {"$set": {"id": contiguous_id}}
        )
    if DEBUG: print(f'Swap took {time.time() - start}') 

def map_contiguous_ids2(collection_bin, DEBUG=False):
    if DEBUG: start = time.time()
    
    # Get distinct sample IDs and sort them using aggregation pipeline
    pipeline = [
        {"$group": {"_id": "$id"}},
        {"$sort": {"_id": 1}}
    ]
    sorted_ids = [doc["_id"] for doc in collection_bin.aggregate(pipeline)]
    
    # Create a dictionary to map original sample IDs to contiguous IDs
    id_map = {id: contiguous_id for contiguous_id, id in enumerate(sorted_ids)}
    
    # Update documents in bulk using bulk_write()
    bulk_operations = []
    for original_id, contiguous_id in id_map.items():
        bulk_operations.append(
            UpdateMany(
                {"id": original_id},
                {"$set": {"id": contiguous_id}}
            )
        )
    collection_bin.bulk_write(bulk_operations)
    
    if DEBUG: print(f'Swap took {time.time() - start}')


def update_chunk(chunk, db_name, collection_name):
    client = MongoClient('mongodb://arctrdcn018:27017/')  # Establish a new MongoDB connection for each worker process
    db = client['wirehead_test']
    collection = db['read']['bin']
    
    bulk_operations = []
    for original_id, contiguous_id in chunk:
        bulk_operations.append(
            UpdateMany(
                {"id": original_id},
                {"$set": {"id": contiguous_id}}
            )
        )
    collection.bulk_write(bulk_operations)
    
    client.close()  # Close the MongoDB connection when done

def map_contiguous_ids3(collection_bin, num_processes=10, DEBUG=False):
    if DEBUG: start = time.time()
    
    # Get distinct sample IDs and sort them using aggregation pipeline
    pipeline = [
        {"$group": {"_id": "$id"}},
        {"$sort": {"_id": 1}}
    ]
    sorted_ids = [doc["_id"] for doc in collection_bin.aggregate(pipeline)]
    
    # Create a dictionary to map original sample IDs to contiguous IDs
    id_map = {id: contiguous_id for contiguous_id, id in enumerate(sorted_ids)}
    
    # Split the id_map into chunks for parallel processing
    chunk_size = math.ceil(len(id_map) / num_processes)
    chunks = [list(id_map.items())[i:i+chunk_size] for i in range(0, len(id_map), chunk_size)]
    
    # Get the database and collection names
    db_name = collection_bin.database.name
    collection_name = collection_bin.name
    
    # Create a multiprocessing pool and update documents in parallel
    with Pool(processes=num_processes) as pool:
        pool.starmap(update_chunk, [(chunk, db_name, collection_name) for chunk in chunks])
    
    if DEBUG: print(f'Swap took {time.time() - start}')

def reset_capped_collections(db, max_samples = SWAP_THRESHOLD):
    # Check if the collections already exist
    if 'write' in db.list_collection_names():
        db.drop_collection('write')
    if 'read' in db.list_collection_names():
        db.drop_collection('read')

    # Create the capped collections with the specified size limit
    create_capped_collection(db, 'read', max_samples=max_samples)
    create_capped_collection(db, 'write', max_samples=max_samples)
    print(f"Capped collections '{'write'}' and '{'read'}' have been reset.")

def get_mongo_bytes(db):
    """Returns the size of mongo read and write halves in bytes"""
    raise NotImplementedError()

def get_mongo_write_size(db):
    """Returns the number of samples in the write half of mongo"""
    raise NotImplementedError()

#############################
### Functions for dataset ###
#############################
def safe_fetch_separate(collection_bin, id_iterator, sample_kinds,
                        nchunks=NUMCHUNKS, max_fetches=10,
                        fetches=0, DEBUG=False) -> (torch.Tensor, torch.Tensor):
    """Safely returns an image-label pair from a MongoDB collection"""
    try:
        sample_id = next(id_iterator)
        
        data_chunks = list(collection_bin.find(
            {
                "id": sample_id,
                "kind": sample_kinds[0]
            },
            {"_id": 0, "chunk": 1}
        ).sort("chunk_id"))
        
        label_chunks = list(collection_bin.find(
            {
                "id": sample_id,
                "kind": sample_kinds[1]
            },
            {"_id": 0, "chunk": 1}
        ).sort("chunk_id"))
        if len(data_chunks) != nchunks or len(label_chunks) != nchunks:
            if fetches < max_fetches:
                return safe_fetch_separate(collection_bin, id_iterator, sample_kinds, nchunks, max_fetches, fetches + 1, DEBUG)
            else:
                return DEFAULT_IMG, DEFAULT_LAB
        data_binary = b"".join(chunk["chunk"] for chunk in data_chunks)
        label_binary = b"".join(chunk["chunk"] for chunk in label_chunks)
        img_tensor = bin2tensor(data_binary)
        lab_tensor = bin2tensor(label_binary)
        if DEBUG: print(f"{time.time()} {sample_id}")
        return img_tensor, lab_tensor

    except StopIteration:
        # Handle the case when the iterator is exhausted
        return DEFAULT_IMG, DEFAULT_LAB
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

