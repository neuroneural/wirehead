 import torch
from torch.utils.data import DataLoader, Dataset
import time
import redis
import numpy as np
import pickle
import sys

DEFAULT_HOST = 'arctrdagn019'
DEFAULT_PORT = 6379
DEFAULT_DBKEY = 'db0'
ERROR_STRING = """Oppsie Woopsie! Uwu Redwis made a shwuky wucky!! A widdle
bwucko boingo! The code monkeys at our headquarters are
working VEWY HAWD to fix this!"""


def get_queue_len(host = DEFAULT_HOST, port = DEFAULT_PORT):
    try:
        r = redis.Redis(host= DEFAULT_HOST, port=DEFAULT_PORT, db=0)
        return r.llen('db0'), r.llen('db1')
    except:
        return -1, -1

class wirehead_dataloader(Dataset):
    def __init__(self, host = DEFAULT_HOST, port = DEFAULT_PORT, db_key=DEFAULT_DBKEY):
        self.r = redis.Redis(host=host, port=port)
        self.db_key = 'db0'
    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        while True:
            pickled_data = self.r.lpop(self.db_key)
            if pickled_data is not None:
                self.r.rpush(self.db_key, pickled_data)
                data = pickle.loads(pickled_data)
                return data[0], data[1]
            else:
                time.sleep(0.5)

if __name__ == "__main__":
    lendb0, lendb1 = get_queue_len()
    if lendb0 == 0:
        print('database is kinda empty, please wait')
        exit()
    dataset = wirehead_dataloader()
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        print(batch[0])
        time.sleep(0.5)
        im, lab = [],[]
  
