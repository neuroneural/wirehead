import sys
import numpy as np
from wirehead import WireheadManager, WireheadGenerator

# Synthseg config
WIREHEAD_CONFIG     = "config.yaml"
PATH_TO_DATA        = "./SynthSeg/data/training_label_maps/"
DATA_FILES          = [f"training_seg_{i:02d}.nii.gz" for i in range(1, 21)]
PATH_TO_SYNTHSEG    = './SynthSeg'

LABEL_MAP = np.asarray(
    [0, 0, 1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 9, 10]
    + [11, 12, 13, 14, 15]
    + [0] * 6
    + [1, 16, 0, 17]
    + [0] * 12
    + [18, 19, 20, 21, 0, 22, 23]
    + [0, 24, 25, 26, 27, 28, 29, 0, 0, 18, 30, 0, 31]
    + [0] * 75
    + [3, 4]
    + [0] * 25
    + [20, 21]
    + [0] * 366,
    dtype="int",
).astype(np.uint8)

def merge_homologs(label):#, device):
    max_value = 31
    # idx = torch.arange(max_value + 1, dtype=torch.long).to(device)
    idx = np.arange(max_value+1)
    idx[31] = 17
    idx[30] = 16
    idx[29] = 15
    idx[28] = 14
    idx[27] = 10
    idx[26] = 9
    idx[25] = 8
    idx[24] = 7
    idx[23] = 6
    idx[22] = 5
    idx[21] = 4
    idx[20] = 3
    idx[19] = 2
    idx[18] = 1
    # return the corresponding values from idx
    return idx[label]

def preprocess_label(lab, label_map=LABEL_MAP):
    return label_map[lab].astype(np.uint8)

def preprocess_image_min_max(img: np.ndarray):
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img

def preprocessing_pipe(data):
    """ Set up your preprocessing options here, ignore if none are needed """
    img, lab = data
    img = preprocess_image_min_max(img) * 255
    img = img.astype(np.uint8)
    lab = preprocess_label(lab)
    lab = merge_homologs(lab)
    lab = lab.astype(np.uint8)    
    return (img, lab) 

def hardware_setup():
    """ Clean slate to set up your hardware, ignore if none are needed """
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    sys.path.append(PATH_TO_SYNTHSEG)
    pass

# Create a generator function that yields desired samples
def create_generator(task_id = 0, training_seg=None):
    """ Creates an iterator that returns data for mongo.
        Should contain all the dependencies of the brain generator
        Preprocessing should be applied at this phase 
        yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str)) """

    # 0. Optionally set up hardware configs
    hardware_setup()
    # 1. Declare your generator and its dependencies here
    sys.path.append(PATH_TO_SYNTHSEG)
    from SynthSeg.brain_generator import BrainGenerator
    training_seg = DATA_FILES[task_id % len(DATA_FILES)] if training_seg == None else training_seg
    brain_generator = BrainGenerator(PATH_TO_DATA + training_seg)
    print(f"Generator: SynthSeg is generating off {training_seg}",flush=True,)
    # 2. Run your generator in a loop, and pass in your preprocessing options
    while True:
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        # 3. Yield your data, which will automatically be pushed to mongo
        yield (img, lab)

if __name__ == "__main__":
    brain_generator    = create_generator()
    wirehead_generator = WireheadGenerator(generator = brain_generator, config_path = WIREHEAD_CONFIG)
    wirehead_generator.run_generator()
