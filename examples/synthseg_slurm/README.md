# Wirehead + Synthseg Local Example

This folder contains instructions to install, run and customize [SynthSeg](https://github.com/BBillot/SynthSeg) + Wirehead. 

---

## Installation 

[Install wirehead and mongoDB](https://github.com/neuroneural/wirehead/blob/main/README.md)

Install the SynthSeg generator
```
git clone https://github.com/neuronets/nobrainer.git
cd nobrainer
git checkout synthseg
pip install -e .
```


All proceeding instructions are executed in examples/synthseg_slurm (where this README lives)

Download an example image
```
curl -L https://github.com/neuroneural/nobrainer-synthseg/raw/master/data/example.nii.gz -o ./example.nii.gz
```

Update the config.yaml to point to your MongoDB instance
```
MONGOHOST: << your host name or IP >>       ; example : "localhost", "0.0.0.0"
DBNAME: << whatever db name you'd prefer >> ; example : "synthseg_slurm"
```


Then, on your login node (that is able to run sbatch jobs), run the test
```
chmod +x test.sh
./test.sh
```

## Synthseg specifics

For this example, we use the data provided by [SynthSeg](https://github.com/BBillot/SynthSeg)

To replace the data used for SynthSeg generation, modify the PATH_TO_DATA and DATA_FILES array with the path to your desired .nii.gz files

Preprocessing can be done inside the generator function. See [generator.py](https://github.com/neuroneural/wirehead/blob/doc/examples/synthseg/generator.py) for a more detailed example:

While quantization to smaller dtypes isn't strictly necessary, it is highly recommended.

---

## Contact

If you have any questions specific to the Wirehead pipeline, please raise an issue or contact us at mdoan4@gsu.edu
