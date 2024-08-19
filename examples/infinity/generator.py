import numpy as np
import time
from wirehead import WireheadGenerator
import uuid
import os
import sys

def create_generator():
    img = np.random.rand(256,256,256)
    lab = np.random.rand(256,256,256)
    while True:
        yield (img, lab)

class OutputCapture:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    # Specify the directory to save the output file
    output_directory = "logs/100/"
    
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate a unique filename using UUID
    filename = os.path.join(output_directory, f"{uuid.uuid4()}.txt")
    
    # Redirect stdout to both console and file
    sys.stdout = OutputCapture(filename)
    
    brain_generator = create_generator()
    wirehead_runtime = WireheadGenerator(
        generator = brain_generator,
        config_path = "config.yaml",
    )
    
    print(f"Logging output to: {filename}")
    wirehead_runtime.run(verbose=True)
    
    # Restore original stdout
    sys.stdout = sys.__stdout__
