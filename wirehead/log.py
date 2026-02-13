""" Wirehead logging types and logger daemon """

import os
import sys
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List

import yaml
from pymongo import MongoClient, DESCENDING


@dataclass
class SwapLog:
    """
    Represents a single swap event in the wirehead system.

    Fields:
        swap_timestamp: unix time when the swap occurred
        prev_swap_timestamp: unix time of the previous swap (None for first)
        swap_duration: seconds the swapped-out cache was live
        swap_count: total number of swaps observed so far
        db_name: name of the mongodb database
    """
    swap_timestamp: float
    prev_swap_timestamp: Optional[float] = None
    swap_duration: Optional[float] = None
    swap_count: int = 0
    db_name: str = ""

    def __str__(self):
        duration_str = f"{self.swap_duration:.2f}s" if self.swap_duration else "N/A"
        return (
            f"[swap #{self.swap_count}] "
            f"t={self.swap_timestamp:.3f} "
            f"cache_lifetime={duration_str} "
            f"db={self.db_name}"
        )


class WireheadLogger:
    """
    Daemon that tails the status collection and emits SwapLog entries.
    Polls for new swap events and computes cache lifetimes.
    """

    def __init__(self, config_path: str, poll_interval: float = 2.0):
        if not os.path.exists(config_path):
            print(f"Logger: config not found at {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        dbname = config.get("DBNAME")
        mongohost = config.get("MONGOHOST")
        port = config.get("PORT", 27017)
        client = MongoClient(f"mongodb://{mongohost}:{port}")

        self.db = client[dbname]
        self.db_name = dbname
        self.status = self.db["status"]
        self.poll_interval = poll_interval
        self.swap_count = 0
        self.last_seen_id = None
        self.prev_swap_timestamp = None

    def _fetch_new_swaps(self) -> list:
        """Fetch status docs newer than last_seen_id, ordered oldest-first."""
        query = {"swapped": True}
        if self.last_seen_id is not None:
            query["_id"] = {"$gt": self.last_seen_id}
        cursor = self.status.find(query).sort("_id", 1)
        return list(cursor)

    def _process_doc(self, doc) -> SwapLog:
        """Convert a status document into a SwapLog."""
        ts = doc.get("swap_timestamp")
        if ts is None:
            # Fall back to ObjectId embedded timestamp
            ts = doc["_id"].generation_time.timestamp()

        duration = None
        if self.prev_swap_timestamp is not None:
            duration = ts - self.prev_swap_timestamp

        self.swap_count += 1
        log = SwapLog(
            swap_timestamp=ts,
            prev_swap_timestamp=self.prev_swap_timestamp,
            swap_duration=duration,
            swap_count=self.swap_count,
            db_name=self.db_name,
        )
        self.prev_swap_timestamp = ts
        self.last_seen_id = doc["_id"]
        return log

    def history(self) -> List[SwapLog]:
        """Return SwapLogs for all past swaps already in the status collection."""
        docs = list(self.status.find({"swapped": True}).sort("_id", 1))
        logs = []
        for doc in docs:
            logs.append(self._process_doc(doc))
        return logs

    def run(self):
        """Poll for new swap events and print SwapLogs."""
        print(f"Logger: watching {self.db_name} (poll={self.poll_interval}s)")

        # Replay history first
        for log in self.history():
            print(log)

        # Then tail for new events
        while True:
            new_docs = self._fetch_new_swaps()
            for doc in new_docs:
                log = self._process_doc(doc)
                print(log)
            time.sleep(self.poll_interval)


def main():
    parser = argparse.ArgumentParser(
        prog="wirehead-logger",
        description="Wirehead swap logger daemon",
    )
    parser.add_argument(
        "config",
        help="path to wirehead config.yaml",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="seconds between polls (default: 2.0)",
    )
    args = parser.parse_args()
    logger = WireheadLogger(args.config, poll_interval=args.poll_interval)
    logger.run()


if __name__ == "__main__":
    main()
