from wirehead import WireheadManager, WireheadGenerator

if __name__ == "__main__":
    wirehead_manager = WireheadManager(config_path = "./conf/wirehead_config.yaml")
    wirehead_manager.run_manager()
