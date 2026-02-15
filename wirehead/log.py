""" Wirehead logging types and logger daemon """

import curses
import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, asdict, field
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

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))

    def __str__(self):
        return self.to_jsonl()


@dataclass
class WireheadSnapshot:
    """Point-in-time snapshot of the full wirehead system state."""
    # generator progress
    samples_started: int = 0
    samples_completed: int = 0
    in_flight: int = 0
    swap_cap: int = 0
    fill_pct: float = 0.0

    # swap lock
    is_swapping: bool = False
    swap_lock_duration: Optional[float] = None

    # swap history
    swap_count: int = 0
    swap_timestamp: Optional[float] = None
    swap_duration: Optional[float] = None
    time_since_last_swap: Optional[float] = None
    avg_swap_duration: Optional[float] = None

    # collection stats
    write_docs: int = 0
    write_size_mb: float = 0.0
    read_docs: int = 0
    read_size_mb: float = 0.0

    # mongo server
    mongo_connections: int = 0

    # metadata
    db_name: str = ""
    poll_interval: float = 2.0
    timestamp: float = 0.0

    # recent swaps for TUI display
    recent_swaps: List[SwapLog] = field(default_factory=list)

    def to_jsonl(self) -> str:
        d = asdict(self)
        # Serialize recent_swaps as list of dicts
        return json.dumps(d)

    def __str__(self):
        return self.to_jsonl()


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
        self.client = MongoClient(f"mongodb://{mongohost}:{port}")

        self.db = self.client[dbname]
        self.db_name = dbname
        self.status = self.db["status"]
        self.poll_interval = poll_interval
        self.swap_count = 0
        self.last_seen_id = None
        self.prev_swap_timestamp = None

        # Config values for metrics
        self.swap_cap = config.get("SWAP_CAP", 0)
        self.counter_collection = config.get("COUNTER_COLLECTION", "counter")
        self.write_collection = config.get("WRITE_COLLECTION", "write") + ".bin"
        self.read_collection = config.get("READ_COLLECTION", "read") + ".bin"

        # Track swap history for rolling averages
        self.swap_logs: List[SwapLog] = []

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
        self.swap_logs.append(log)
        return log

    def history(self) -> List[SwapLog]:
        """Return SwapLogs for all past swaps already in the status collection."""
        docs = list(self.status.find({"swapped": True}).sort("_id", 1))
        logs = []
        for doc in docs:
            logs.append(self._process_doc(doc))
        return logs

    def _get_counter(self, field: str) -> int:
        """Read a counter value from the counter collection."""
        doc = self.db[self.counter_collection].find_one({"_id": field})
        if doc is None:
            return 0
        return doc.get("sequence_value", 0)

    def _get_coll_stats(self, coll_name: str) -> tuple:
        """Return (doc_count, size_mb) for a collection."""
        try:
            stats = self.db.command("collStats", coll_name)
            count = stats.get("count", 0)
            size_mb = stats.get("size", 0) / (1024 * 1024)
            return count, size_mb
        except Exception:
            return 0, 0.0

    def _get_swap_lock(self) -> tuple:
        """Return (is_locked, duration_if_locked)."""
        doc = self.db[self.counter_collection].find_one({"_id": "swap_lock"})
        if doc is None or not doc.get("locked", False):
            return False, None
        ts = doc.get("timestamp")
        if ts is not None:
            return True, time.time() - ts
        return True, None

    def snapshot(self) -> WireheadSnapshot:
        """Gather all system metrics into a single snapshot."""
        now = time.time()

        # Process any new swap events
        new_docs = self._fetch_new_swaps()
        for doc in new_docs:
            self._process_doc(doc)

        # Counter collection
        started = self._get_counter("started")
        completed = self._get_counter("completed")
        in_flight = started - completed
        fill_pct = (completed / self.swap_cap * 100) if self.swap_cap > 0 else 0.0

        # Swap lock
        is_swapping, swap_lock_duration = self._get_swap_lock()

        # Swap history metrics
        swap_ts = self.swap_logs[-1].swap_timestamp if self.swap_logs else None
        swap_dur = self.swap_logs[-1].swap_duration if self.swap_logs else None
        time_since = (now - swap_ts) if swap_ts is not None else None

        durations = [s.swap_duration for s in self.swap_logs if s.swap_duration is not None]
        avg_dur = sum(durations) / len(durations) if durations else None

        # Collection stats
        write_docs, write_size_mb = self._get_coll_stats(self.write_collection)
        read_docs, read_size_mb = self._get_coll_stats(self.read_collection)

        # Mongo connections
        mongo_connections = 0
        try:
            server_status = self.db.command("serverStatus")
            mongo_connections = server_status.get("connections", {}).get("current", 0)
        except Exception:
            pass

        return WireheadSnapshot(
            samples_started=started,
            samples_completed=completed,
            in_flight=in_flight,
            swap_cap=self.swap_cap,
            fill_pct=fill_pct,
            is_swapping=is_swapping,
            swap_lock_duration=swap_lock_duration,
            swap_count=self.swap_count,
            swap_timestamp=swap_ts,
            swap_duration=swap_dur,
            time_since_last_swap=time_since,
            avg_swap_duration=avg_dur,
            write_docs=write_docs,
            write_size_mb=write_size_mb,
            read_docs=read_docs,
            read_size_mb=read_size_mb,
            mongo_connections=mongo_connections,
            db_name=self.db_name,
            poll_interval=self.poll_interval,
            timestamp=now,
            recent_swaps=list(self.swap_logs[-10:]),
        )

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

    def run_daemon(self):
        """Continuously emit full system snapshots as JSONL."""
        # Bootstrap swap history
        self.history()

        while True:
            snap = self.snapshot()
            print(snap.to_jsonl(), flush=True)
            time.sleep(self.poll_interval)

    def run_tui(self):
        """Launch the curses TUI dashboard."""
        # Bootstrap swap history before entering curses
        self.history()
        curses.wrapper(self._tui_main)

    def _tui_main(self, stdscr):
        """Curses main loop."""
        curses.curs_set(0)
        stdscr.timeout(int(self.poll_interval * 1000))
        curses.use_default_colors()

        while True:
            snap = self.snapshot()
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            self._draw_tui(stdscr, snap, width, height)
            stdscr.refresh()

            key = stdscr.getch()
            if key == ord('q'):
                break

    def _draw_tui(self, stdscr, snap: WireheadSnapshot, width: int, height: int):
        """Render the TUI dashboard."""
        row = 0
        w = max(width, 60)

        def put(y, x, text, attr=0):
            if y < height:
                stdscr.addnstr(y, x, text, w - x - 1, attr)

        # Header
        header = f"wirehead — {snap.db_name}"
        poll_tag = f"[poll: {snap.poll_interval}s]"
        put(row, 0, header, curses.A_BOLD)
        put(row, max(0, w - len(poll_tag) - 1), poll_tag)
        row += 1
        put(row, 0, "─" * (w - 1))
        row += 1

        # Generator + Swap Status (side by side)
        put(row, 1, "GENERATOR", curses.A_BOLD)
        put(row, 28, "SWAP STATUS", curses.A_BOLD)
        row += 1

        cap = snap.swap_cap
        put(row, 1, f"started:    {snap.samples_started}/{cap}")
        put(row, 28, f"total swaps:  {snap.swap_count}")
        row += 1

        put(row, 1, f"completed:  {snap.samples_completed}/{cap}")
        ts_str = f"{snap.time_since_last_swap:.0f}s ago" if snap.time_since_last_swap is not None else "—"
        put(row, 28, f"last swap:    {ts_str}")
        row += 1

        put(row, 1, f"in flight:  {snap.in_flight}")
        avg_str = f"{snap.avg_swap_duration:.1f}s" if snap.avg_swap_duration is not None else "—"
        put(row, 28, f"avg lifetime: {avg_str}")
        row += 1

        # Fill bar
        bar_width = 20
        filled = int(snap.fill_pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        put(row, 1, f"fill:       [{bar}] {snap.fill_pct:.0f}%")

        if snap.is_swapping:
            lock_str = f"SWAPPING ({snap.swap_lock_duration:.1f}s)" if snap.swap_lock_duration else "SWAPPING"
            put(row, 28, f"state: {lock_str}", curses.A_BOLD)
        else:
            put(row, 28, "state: IDLE")
        row += 1

        put(row, 0, "─" * (w - 1))
        row += 1

        # Collections + Mongo
        put(row, 1, "COLLECTIONS", curses.A_BOLD)
        put(row, 40, "MONGO", curses.A_BOLD)
        row += 1

        put(row, 1, f"{snap.write_docs:>6} docs  ({snap.write_size_mb:.1f} MB)  write")
        put(row, 40, f"connections: {snap.mongo_connections}")
        row += 1

        put(row, 1, f"{snap.read_docs:>6} docs  ({snap.read_size_mb:.1f} MB)  read")
        row += 1

        put(row, 0, "─" * (w - 1))
        row += 1

        # Recent Swaps
        put(row, 1, "RECENT SWAPS", curses.A_BOLD)
        row += 1

        recent = list(reversed(snap.recent_swaps[-5:]))
        if not recent:
            put(row, 1, "(none)")
            row += 1
        for s in recent:
            if row >= height - 2:
                break
            dur_str = f"lifetime={s.swap_duration:.2f}s" if s.swap_duration is not None else "lifetime=—"
            put(row, 1, f"#{s.swap_count:<4} {s.swap_timestamp:.3f}  {dur_str}")
            row += 1

        put(row, 0, "─" * (w - 1))
        row += 1

        # Footer
        if row < height:
            quit_msg = "q to quit"
            put(row, max(0, w - len(quit_msg) - 1), quit_msg)


def main():
    parser = argparse.ArgumentParser(
        prog="wirehead",
        description="Wirehead CLI",
    )
    parser.add_argument(
        "config",
        help="path to wirehead config.yaml",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="run continuous JSONL snapshot stream",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="launch curses TUI dashboard",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="seconds between polls (default: 2.0)",
    )
    args = parser.parse_args()

    logger = WireheadLogger(args.config, poll_interval=args.poll_interval)

    if args.tui:
        logger.run_tui()
    elif args.daemon:
        logger.run_daemon()
    else:
        # Default: print swap history as JSONL and exit
        logs = logger.history()
        if not logs:
            print("No swaps recorded yet.")
        for log in logs:
            print(log)


if __name__ == "__main__":
    main()
