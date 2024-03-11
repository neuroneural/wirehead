from pymongo import MongoClient
from wirehead import functions, defaults
import torch


if __name__ == "__main__":
    db = MongoClient(defaults.MONGO_CLIENT)[defaults.MONGO_DBNAME]
    collection_bin_read = db['read']
    collection_bin_write = db['write']
    
    start_id = 0
    end_id = 999
    
    for package_id in range(start_id, end_id):
        # Generate and serialize the image and label tensors
        img = torch.randint(0, 10, (256, 256, 256), dtype=torch.uint8)
        lab = torch.randint(0, 10, (256, 256, 256), dtype=torch.uint8)
        
        # Create the package tuple
        package = (img, lab)
        # Push the package to the write collection
        functions.push_mongo(package, package_id, collection_bin_write)


