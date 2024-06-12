import fcntl
import numpy as np
from wirehead import WireheadManager

WIREHEAD_CONFIG = "config.yaml"
LOCK_FILE = "manager.lock"

if __name__ == "__main__":
    # Acquire a lock to ensure only one instance of the manager is running
    lock_file = open(LOCK_FILE, "w")
    try:
        # Plug into wirehead
        wirehead_runtime = WireheadManager(config_path=WIREHEAD_CONFIG)
        wirehead_runtime.run_manager()
        # Release the lock when the manager finishes or encounters an error
        fcntl.lockf(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    except IOError:
        print("Another instance of the manager is already running.")
        sys.exit(1)

