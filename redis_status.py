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
    args = parser.parse_args()

    host = args.ip if args.ip else wh.DEFAULT_HOST
    port = args.port if args.port else wh.DEFAULT_PORT

    fake_im, fake_lab = wh.load_fake_samples()

    while (True): 
        try:
            r = redis.Redis(host=host, port = port)
            r.rpush('status', bytes(True))
            print("Redis Status: OK")
            break 
        except redis.ConnectionError:
            print(f"Redis is loading database...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Exiting.")
            break
            

