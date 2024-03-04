import io   
import bson
from pymongo import MongoClient
import torch
import numpy as np
import pickle 
import time



def insert_sample(
    image_bytes,
    label_bytes,
    id,
    collection_bin,
    chunkSize=10
    ):

    for chunk in chunk_binobj(image_bytes, id, "image", chunkSize):
        collection_bin.insert_one(chunk)
    for chunk in chunk_binobj(label_bytes, id, "label", chunkSize):
        collection_bin.insert_one(chunk)

def chunk_binobj(tensor_compressed, id, kind, chunksize):
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

if __name__=="__main__":
    client = MongoClient('mongodb://10.245.12.58:27017/')

    # specify the database and collection
    db = client['wirehead_test']

    write_side = db['write']
    read_side = db['read']

    shape = (256,256,256)
    start_time = time.time()

    write_stats = db.command("collstats", 'write')
    read_stats = db.command("collstats", 'read')

    bytes_per_sample = 256*256*256*2
    # print the size of the collection in bytes
    print(f'The size of the write side is {write_stats["size"]} bytes.')
    print(f'The size of the write side is {read_stats["size"]} bytes.')

    #print(f'Write side: {int(write_stats["size"][0]) / bytes_per_sample} samples')
    print(type(read_side['size']))
    print(type(read_side['size'][0]))

    print(f'Read side: {int(read_side["size"][0]) / bytes_per_sample} samples')



    print(f"Finished, took {time.time() -start_time}")
