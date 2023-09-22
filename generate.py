import redis
import sys
from time import time
import pickle

sys.path.append('/data/users1/mdoan4/synth')
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator


# Connect to your Redis instance.
r = redis.Redis(host='localhost', port=6379, db=0)
while(True):
    start = time() 
    # generate an image from the label map.
    brain_generator = BrainGenerator('../synthseg/data/training_label_maps/training_seg_01.nii.gz')
    im, lab = brain_generator.generate_brain()
    im_bytes = pickle.dumps(im)
    # push into redis queue
    r.rpush('mylist', im_bytes)
    print('----------------------------------')
    print(f"The generation took {time() - start}")

