import redis
import time
import os
import pickle
import sys
import random
import argparse
import wirehead as wh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    parser.add_argument("--cap", help="Max rotating queue length")
    args = parser.parse_args()

    host = args.ip if args.ip else wh.DEFAULT_HOST
    port = args.port if args.port else wh.DEFAULT_PORT
    cap = int(args.cap) if args.cap else wh.DEFAULT_CAP

    r = redis.Redis(host=host, port = port)
    print(f"Manager: Started successfully and is hosted on {host}")
    #wh.hang_until_redis_is_loaded(r)
    # Optional, in production should just append to database
    while True:
        lendb0, lendb1 = wh.get_queue_len(r)
        print(f"Manager: {time.time()} --- Main: {lendb0}, Swap: {lendb1}")
        if lendb0 == -1:
            print("Manager: db0 is empty")
            while lendb0 == -1:
                time.sleep(1)
                lendb0, lendb1 = wh.get_queue_len(r)
                continue
        # Swap databases whenever db1 is full
        # Timeout between checks 
        time.sleep(wh.MANAGER_TIMEOUT)

