import redis
import time
import os
import pickle
import sys
import random
import argparse



# Things that users should change
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 6379
DEFAULT_CAP = 10 
MANAGER_TIMEOUT = 5


def get_queue_len(r):
    try:
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1

def quantize_to_uint8(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val == min_val:
        return np.zeros_like(tensor, dtype='uint8')
    tensor = ((tensor - min_val) / (max_val - min_val) * 255).round()
    return tensor.astype('uint8')

def lock_db(r, lock_name, timeout=10):
    while True:
        if r.setnx(lock_name, 1):
            r.expire(lock_name, timeout)
            return True
        time.sleep(0.1)

def swap_db(r):
    lock_name = 'swap_lock'
    locked = lock_db(r, lock_name = lock_name)
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

def push_db(r, package_bytes):
    lock_name = 'swap_lock'
    locked = lock_db(r, lock_name = lock_name)
    if locked:
        try:
            r.rpush("db1", package_bytes)
            if not r.exists("db0"):
                r.rpush("db0", package_bytes)
        finally:
            r.delete(lock_name)

def hang_until_redis_is_loaded(r):
    while (True):
        try:
            r.rpush('status', bytes(True))
            break
            return
        except redis.ConnectionError:
            print(f"Redis is loading database...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Exiting.")
            break
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="IP address for Redis")
    parser.add_argument("--port", help="Port for Redis")
    parser.add_argument("--cap", help="Max rotating queue length")
    args = parser.parse_args()

    host = args.ip if args.ip else DEFAULT_HOST
    port = args.port if args.port else DEFAULT_PORT
    cap = int(args.cap) if args.cap else DEFAULT_CAP

    r = redis.Redis(host=host, port = port)
    print(f"Manager: Started successfully and is hosted on {host}")
    hang_until_redis_is_loaded(r)
    # Optional, in production should just append to database
    while True:
        lendb0, lendb1 = get_queue_len(r)
        print(time.time(), lendb0, lendb1)
        if lendb0 == -1:
            print("Error: db0 is empty")
            while lendb0 == -1:
                time.sleep(5)
                lendb0, lendb1 = get_queue_len(r)
                continue
        # Swap databases whenever db1 is full
        if lendb1 > cap:
            swap_db(r)
        # Timeout between checks 
        time.sleep(MANAGER_TIMEOUT)

