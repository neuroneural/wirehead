from wirehead import WireheadManager

if __name__ == "__main__":
    wirehead_runtime = WireheadManager(config_path="config.yaml")
    wirehead_runtime.run_manager()
