import pymongo
import threading
import time
from defaults import MONGO_CLIENT
from wirehead import functions, defaults

if __name__ == '__main__':
    # Create a MongoDB client
    db = pymongo.MongoClient(MONGO_CLIENT)['wirehead_test']
    # Start monitoring insertions in the write collection
    insertion_monitor = threading.Thread(target=functions.monitor_insertions, args=(db,True,))
    insertion_monitor.daemon = True
    insertion_monitor.start()  
    while(True):
        time.sleep(10)