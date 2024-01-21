import redis
import time
import os
import pickle
import sys
import random
import argparse

from wirehead_defaults import *
from wirehead_utils import connect_to_redis, get_redis_len, lock_redis

def swap_db(r):
    lock_name = 'swap_lock'
    locked = lock_redis(r, lock_name = lock_name)
    if locked:
        try:
            pipe = r.pipeline()
            if r.exists("db0") and r.exists("db1"):
                pipe.multi()
                pipe.rename("db0", "temp_key")
                pipe.rename("db1", "db0")
                pipe.rename("temp_key", "db1")
                pipe.delete('db1')
                pipe.execute()
        finally:
            r.delete(lock_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    parser.add_argument("--cap", help="Max rotating queue length")
    args = parser.parse_args()

    host = args.ip if args.ip else DEFAULT_HOST
    port = args.port if args.port else DEFAULT_PORT
    cap = int(args.cap) if args.cap else DEFAULT_CAP

    r = connect_to_redis(host, port) 
    while True:
        lendb0, lendb1 = get_redis_len(r)
        print(time.time(), lendb0, lendb1)
        if lendb0 == -1:
            print("Error: db0 is empty")
            # Hang util redis is alive
            while lendb0 == -1:
                time.sleep(5)
                lendb0, lendb1 = get_redis_len(r)
                continue
        # Swap databases whenever db1 is full
        if lendb1 > cap:
            swap_db(r)
        # Timeout between checks 
        time.sleep(MANAGER_TIMEOUT)

