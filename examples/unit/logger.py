from wirehead import WireheadLogger

if __name__ == "__main__":
    logger = WireheadLogger("config.yaml")

    # Print swap history
    logs = logger.history()
    if not logs:
        print("No swaps recorded yet.")
    for log in logs:
        print(log)

    # Run as daemon (tails for new swaps)
    # logger.run()
