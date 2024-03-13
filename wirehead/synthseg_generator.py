# Creates a Synthseg brain generator and continuously pushees to MongoDB 
import sys
import random
import argparse
from time import time

import torch
import numpy as np
from pymongo import MongoClient
from wirehead import defaults, functions
### Synthseg imports ###
sys.path.append(defaults.SYNTHSEG_PATH)
from SynthSeg.brain_generator import BrainGenerator

def preprocess_synthseg_label(lab: np.ndarray) -> np.ndarray:
    """Converst a label map from synthseg to a contiguous mapping 0..255"""
    synthseg_label_map = np.asarray([ 0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 1, 16, 0, 17, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 19, 20, 21, 0, 22, 23, 0,
        24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0], dtype='int').astype(np.uint8)
    return synthseg_label_map[lab].astype(np.uint8)

def generation_loop(collection_bin, generator, id_range, DEBUG=False):
    """Preprocessing and pushing of samples from synthseg"""
    idx = functions.gen_id_iterator(id_range)
    while(True):
        if DEBUG:
            total_time = time() 
            generation_start = time()
        img, lab = generator.generate_brain()
        if DEBUG: 
            generation_end = time() 

        if DEBUG: 
            preprocess_start = time()
        img = functions.preprocess_image_min_max(img).astype(np.uint8) * 255 # Normalize in 0..255
        lab = preprocess_synthseg_label(lab) # Convert to contiguous 0..255
        img = torch.from_numpy(img)
        lab = torch.from_numpy(lab)
        if DEBUG: 
            preprocess_end = time()
        functions.push_mongo((img, lab), next(idx), collection_bin)
        if DEBUG:
            print(f'Generation took {generation_end - generation_start} seconds')
            print(f'Preprocessing took {preprocess_end - preprocess_start} seconds')
            print(f'In total, process took {time() - total_time} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client',     help = "Mongo client hosting wirehead")
    parser.add_argument('--train_seg',  help = "Segment to use for generator")
    parser.add_argument('--id_start',   help = "Start of generator ID range")
    parser.add_argument('--id_end',     help = "End of generator ID range")

    args        = parser.parse_args()
    client_name = args.client if args.client else defaults.MONGO_CLIENT
    train_seg   = args.training_seg if args.train_seg else random.choice(defaults.DATA_FILES)
    id_start    = args.id_start if args.id_start else 0
    id_end      = args.id_end if args.id_end else defaults.SWAP_THRESHOLD

    client = MongoClient(client_name)    
    db = client[defaults.MONGO_DBNAME]
    write_col = db['write']['bin']
    generator = BrainGenerator(defaults.DATA_PATH+ train_seg)
    generation_loop(write_col, generator, (id_start, id_end), DEBUG=True)