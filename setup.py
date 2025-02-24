from setuptools import setup, find_packages

setup(
    name='wirehead',
    version='0.9.1',
    packages=find_packages(),
    description="Caching system for scaling of synthetic data generators using MongoDB",
    long_description="""# Wirehead

Caching system for scaling of synthetic data generators using MongoDB.

## Features
- Cache and efficiently serve synthetic data from generators
- Scalable architecture using MongoDB for storage
- Support for numpy and torch tensors
- Configurable caching behavior

## Quick Start
```python
# Generator example
import numpy as np
from wirehead import WireheadGenerator 

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
wirehead_runtime.run_generator()

# Dataset example
from wirehead import MongoheadDataset
dataset = MongoheadDataset(config_path = "config.yaml")
data = dataset[[0]]
```

## MongoDB Setup Required
Requires a running MongoDB instance.

## Documentation
For full documentation and examples, visit: https://github.com/neuroneural/wirehead
""",
    long_description_content_type="text/markdown",
    author="Neuroneural Lab",
    author_email="mdoan4@gsu.edu",
    url="https://github.com/neuroneural/wirehead",
    install_requires=[
        'pymongo',
        'torch',
        'numpy',
        'PyYaml',
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line entry points here
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
