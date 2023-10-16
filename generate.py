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

    fake_im, fake_lab = map(wh.quantize_to_uint8, wh.load_fake_samples())


    while (True): 
        try:
            r = redis.Redis(host=host, port = port)
            #wh.hang_until_redis_is_loaded(r)
            print(f"Generator: Connecting to {host}")

            while(True):
                im, lab = fake_im, fake_lab 
                package = (im,lab)
                package_bytes = pickle.dumps(package)

                # Push to db1
                wh.push_db(r, package_bytes) 

                #Optional timeout
                time.sleep(0.1)
        except redis.ConnectionError:
            print(f"Generator: Failed to connect to redis server on {host}, retrying...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Exiting.")
            break
            

