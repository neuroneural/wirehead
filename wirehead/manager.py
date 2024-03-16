import pymongo
import threading
import time
import random
from defaults import MONGO_CLIENT
from wirehead import functions, defaults

def monitor_insertions(db, DEBUG=False):
    """ Monitors the database in a separate thread, swapping the databases
        when samples are full"""
    global total_records
    total_records = 0
    read_count = 0
    write_collection = db['write']['bin']

    while True:
        write_count = len(write_collection.distinct('id'))
        print(f"Write: {write_count}, Read: {read_count}, Time: {time.time()}")
        if write_count>= defaults.SWAP_THRESHOLD:
            functions.swap_db(db, DEBUG=DEBUG)
            total_records += write_count
            print(f'Total records seen: {total_records}')
            read_count = write_count
        time.sleep(10)  # Adjust the polling interval as needed


if __name__ == '__main__':
    # Create a MongoDB client
    db = pymongo.MongoClient(MONGO_CLIENT)['wirehead_test']
    # Start monitoring insertions in the write collection
    insertion_monitor = threading.Thread(target=monitor_insertions, args=(db,True,))
    insertion_monitor.daemon = True
    insertion_monitor.start()  
    while(True):
        time.sleep(10)