import numpy as np
from wirehead import WireheadManager

WIREHEAD_CONFIG = "config.yaml"

if __name__ == "__main__":
    wirehead_runtime = WireheadManager(config_path=WIREHEAD_CONFIG)
    wirehead_runtime.run_manager()
