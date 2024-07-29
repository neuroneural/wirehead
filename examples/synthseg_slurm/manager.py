from wirehead import WireheadManager

WIREHEAD_CONFIG     = "config.yaml"

if __name__ == "__main__":
    wirehead_manager = WireheadManager(
        config_path = WIREHEAD_CONFIG
    )
    wirehead_manager.run_manager()
