# Wirehead + Synthseg Local Example

This folder contains instructions to install, run and customize [SynthSeg](https://github.com/BBillot/SynthSeg) + Wirehead. 

This is specific to running SynthSeg and Wirehead locally. For instructions to run SynthSeg + Wirehead on a slurm cluster, do check out examples/synthseg_slurm

---

## Installation 

These instructions are different from the regular wirehead installation instructions due to the specific dependencies of SynthSeg

Instructions:
```
git clone git@github.com:neuroneural/wirehead.git
cd wirehead/examples/synthseg
```

Install python3.8 and create an environment
```
sudo apt install python3.8 python3.8-venv
python3.8 -m venv wirehead 
source wirehead/bin/activate
```

Install wirehead
```
pip install -e ../../
pip install -r requirements.txt
```

Fetch the data from SynthSeg
```
git clone git@github.com:BBillot/SynthSeg.git
```

Run the test
```
chmod +x test.sh
./test.sh
```

---

## Config guide

All wirehead configs live inside yaml files, and must be specified when declaring wirehead manager, generator and dataset objects. For the system to work, all components must use the __same__ configs.

Basic configs:
```
MONGOHOST -- IP address or hostname for machine running MongoDB instance
DBNAME -- MongoDB database name
PORT -- Port for MongoDB instance. Defaults to 27017
SWAP_CAP -- Size cap for read and write collections. bigger means bigger cache, and less frequent swaps. For synthseg data, a swap_cap of 100 will mean wirehead will use 100 * (256*256*256) * 2 * 2 = 6.25 gigabytes at peak
```

Advanced configs:
```
SAMPLE -- Array of strings denoting name of samples in data tuple. 
WRITE_COLLECTION   -- Name of write collection (generators push to this)
READ_COLLECTION    -- Name of read colletion (dataset reads from this)
COUNTER_COLLECTION -- Name of counter collection for manager metrics
TEMP_COLLECTION    -- Name of temporary collection used for moving data during swap
CHUNKSIZE          -- Number of megabytes used for chunking data
```

---

## Generator guide

```
wirehead_generator = WireheadGenerator(
    generator = generator,
    config_path="config.yaml"
)
wirehead_generator.run_generator()
```

Wirehead's WireheadGenerator object takes in a generator, which is a python generator function. This function yields a tuple containing numpy arrays. The number of samples in this tuple should match the number of strings  specified in SAMPLE in config.yaml

Example:

config.yaml:
```
SAMPLE: ["input", "label"]
```

generating script:
```
def create_generator():
    while True: 
        img = np.random.rand(256,256,256)
        lab = np.random.rand(256,256,256)
        yield (img, lab)

brain_generator = create_generator()
wirehead_runtime = WireheadGenerator(
    generator = brain_generator,
    config_path = "config.yaml" 
)
wirehead_runtime.run_generator() an infinite loop
```

---

## Synthseg specifics

For this example, we use the data provided by [SynthSeg](https://github.com/BBillot/SynthSeg), which live in:
```
SynthSeg/data/training_label_maps/
```

To replace the data used for SynthSeg generation, modify the PATH_TO_DATA and DATA_FILES array with the path to your desired .nii.gz files

Preprocessing can be done inside the generator function. See [generator.py](https://github.com/neuroneural/wirehead/blob/doc/examples/synthseg/generator.py) for a more detailed example:


While quantization to smaller dtypes isn't strictly necessary, it is highly recommended.

```
def preprocessing_pipe(data):
    """ Set up your preprocessing options here, ignore if none are needed """
    img, lab = data
    img = preprocess_image_min_max(img) * 255 # min max normalization
    img = img.astype(np.uint8)  # quantization to uint8
    lab = preprocess_label(lab) # label map conversion
    lab = merge_homologs(lab)   
    lab = lab.astype(np.uint8)  # quantization to uint8  
    return (img, lab) 

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
```

---

## Contact

If you have any questions specific to the Wirehead pipeline, please raise an issue or contact us at mdoan4@gsu.edu
