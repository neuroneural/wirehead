import threading
import numpy as np
from wirehead import Runtime
import fcntl
import sys

WIREHEAD_CONFIG = "config.yaml"
LOCK_FILE = "manager.lock"

def create_generator():
    while True:
        img = np.random.rand(256, 256, 256)
        lab = np.random.rand(256, 256, 256)
        yield (img, lab)

def acquire_lock():
    lock_file = open(LOCK_FILE, "w")
    try:
        fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file
    except IOError:
        print("Another instance of the manager is already running.")
        sys.exit(1)


if __name__ == "__main__":
    # Acquire a lock to ensure only one instance of the manager is running
    lock_file = acquire_lock()

    try:
        # Plug into wirehead
        brain_generator = create_generator()
        wirehead_runtime = Runtime(
            generator=brain_generator,  # Specify generator
            config_path=WIREHEAD_CONFIG  # Specify config
        )
        wirehead_runtime.run_manager()
    finally:
        # Release the lock when the manager finishes or encounters an error
        fcntl.lockf(lock_file, fcntl.LOCK_UN)
        lock_file.close()
