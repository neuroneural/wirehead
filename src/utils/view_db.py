import redis
import time
import os
import pickle
import sys
import random
import argparse

from wirehead_utils import connect_to_redis, get_redis_len
from wirehead_defaults import *

if __name__ == '__main__':
    """
    This script will simply read the current status
    of a hosted wirehead server without making modifications
    to the server. Safe to use as a monitoring tool.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    args = parser.parse_args()

    host = args.ip if args.ip else DEFAULT_HOST
    port = args.port if args.port else DEFAULT_PORT

    r = connect_to_redis(host,port) 

    while True:
        lendb0, lendb1 = get_redis_len(r)
        print(f"View: {time.time()} --- Main: {lendb0}, Swap: {lendb1}")
        if lendb0 == -1:
            print("View: db0 is empty")
            while lendb0 == -1:
                time.sleep(1)
                lendb0, lendb1 = get_redis_len(r)
                continue
        time.sleep(MANAGER_TIMEOUT)

