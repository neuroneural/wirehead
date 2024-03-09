import os
import sys
import torch
import random
import pickle
import argparse
import numpy as np
from time import time
from pymongo import MongoClient
from wirehead import defaults, functions
### Synthseg imports ###
sys.path.append(defaults.SYNTHSEG_PATH)
from SynthSeg.brain_generator import BrainGenerator
from ext.lab2im import utils

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

def generation_loop(db, generator, gen_len, debug=False):
    """Preprocessing and pushing of samples from synthseg"""
    for _ in range(gen_len):
        total_time          = time() 
        generation_start    = time()
        img, lab = generator.generate_brain()
        generation_end      = time() 

        preprocess_start    = time()
        img = functions.preprocess_image_min_max(img) * 255 # Normalize in 0..255
        img = img.astype(np.uint8) 
        lab = functions.preprocess_label_synthseg(lab) # Convert to contiguous 0..255
        img = functions.tensor2bin(torch.from_numpy(img)) # Convert to serialized tensor, quantized uint8
        lab = functions.tensor2bin(torch.from_numpy(lab)) # Convert to serialized tensor, quantized uint8
        preprocess_end      = time()
        
        pickle_start        = time()
        package = (img, lab) # Package into single tuple for serialization
        package_bytes = pickle.dumps(package)
        pickle_end          = time()
        functions.push_mongo(db, package_bytes)

        if debug:
            print(f'Generation took {generation_end - generation_start}')
            print(f'Preprocessing took {preprocess_end - preprocess_end}')
            print(f'Pickling took {pickle_end - pickle_start}')
            print(f'In total, process took {time() - total_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client',     help = "Mongo client hosting wirehead")
    parser.add_argument('--train_seg',  help = "Segment to use for generator")
    parser.add_argument('--gen_len',    help = "Numbre of samples to generate")

    args = parser.parse_args()
    client = args.client if args.client else defaults.MONGO_CLIENT
    train_seg = args.training_seg if args.train_seg else random.choice(defaults.DATA_FILES)

    gen_len = args.gen_len if args.gen_len else  defaults.GENERATOR_LENGTH

    client = MongoClient(client)    
    db = client[defaults.MONGO_DBNAME]
    generator = BrainGenerator(defaults.PATH_TO_DATA + train_seg)
    generation_loop(db, generator, gen_len, debug=True)


