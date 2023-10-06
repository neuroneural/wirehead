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


    while (True): 
        try:
            r = redis.Redis(host=host, port = port)
            print(f"Connecting to {host}")

            while(True):
                random_number = random.randint(1, 10000)
                im, lab = random_number, random_number
                package = (im,lab)
                package_bytes = pickle.dumps(package)

                # Push to db1
                wh.push_db(r, package_bytes) 

                #Optional timeout
                time.sleep(0.1)
        except redis.ConnectionError:
            print(f"Failed to connect to redis server on {host}, retrying...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Exiting.")
            break
            

