import redis
import time
import os
import pickle
import sys
import random
import argparse

from redishead.defaults import *

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

def connect_to_redis(host, port):
    def hang_until_redis_is_loaded(r):
        while (True):
            try:
                r.rpush('status', bytes(True))
                break
                return 
            except redis.ConnectionError:
                print(f"Generator: Redis is not responding") 
                time.sleep(5)
            except KeyboardInterrupt:
                print("Generator: Terminating at Redis loading.")
                break
                return None
    while(True):
        try:
            r = redis.Redis(host=host, port = port)
            hang_until_redis_is_loaded(r)
            print(f"Generator: Connected to Redis hosted at {host}:{port}")
            return r
        except redis.ConnectionError:
            print(f"Generator: Redis is not responding") 
            time.sleep(5)
        except KeyboardInterrupt:
            print("Generator: Terminating at Redis loading.")
            break
            return None

def get_redis_len(r):
    try:
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1


def lock_redis(r, lock_name, timeout=10):
        """
        This function ensures proper concurrency mangagement
        by redis. Unsafe to edit without extensive testing
        """
        while True:
            if r.setnx(lock_name, 1):
                r.expire(lock_name, timeout)
                return True
            time.sleep(0.1)

def load_fake_samples():
    im = np.load('/data/users1/mdoan4/wirehead/src/samples/image.npy')
    lab = np.load('/data/users1/mdoan4/wirehead/src/samples/label.npy')
    return im, lab



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

