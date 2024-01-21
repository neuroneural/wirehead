import redis
import numpy as np
import time

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

