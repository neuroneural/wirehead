""" Wirehead logging types and logger daemon """

import time
from dataclasses import dataclass, field, asdict
from typing import Optional


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
