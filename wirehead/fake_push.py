from pymongo import MongoClient
from wirehead import functions
from wirehead import defaults
import torch
import io
import pickle


if __name__ == "__main__":
    db = MongoClient(defaults.MONGO_CLIENT)[defaults.MONGO_DBNAME]
    collection_bin_read = db['read']
    collection_bin_write = db['write']
    
    start_id = 0
    end_id = 999
    
    for package_id in range(start_id, end_id):
        # Generate and serialize the image and label tensors
        img_tensor = torch.randint(0, 10, (256, 256, 256), dtype=torch.uint8)
        lab_tensor = torch.randint(0, 10, (256, 256, 256), dtype=torch.uint8)
        
        img_bytes = functions.tensor2bin(img_tensor)
        lab_bytes = functions.tensor2bin(lab_tensor)
        
        # Create the package tuple
        package = (img_bytes, lab_bytes)
        package_bytes = pickle.dumps(package)
        
        # Push the package to the write collection
        functions.push_mongo(package_bytes, package_id, collection_bin_write)


