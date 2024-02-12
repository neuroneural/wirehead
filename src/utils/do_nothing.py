import torch
import redis
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import time


DEFAULT_HOST = 'arctrdcn017' #arctrdcn017
DEFAULT_PORT = 6379
WIREHEAD_NUMSAMPLES=10
def min_max_normalize(x):
    """Min max normalization preprocessing"""
    return (x - x.min()) / (x.max() - x.min())

def quantile_normalize(img, qmin=0.01, qmax=0.99):
    """Unit interval preprocessing"""
    img = (img - img.quantile(qmin)) / (img.quantile(qmax) - img.quantile(qmin))
    return img

def rtransform(x):
    #return min_max_normalize(x)
    return x

def rcollate(batch, size=256):
    '''
    print('ding')
    sample = batch[0][0]
    label = batch[0][1]
    get_tensor_info(sample)
    get_tensor_info(label)
    np.save('./sample.npy', sample)
    np.save('./label.npy', label)
    '''
    data = quantile_normalize(torch.from_numpy(batch[0][0]).float()).unsqueeze(1)
    labels = torch.from_numpy(batch[0][1]).long().unsqueeze(1)
    return data.unsqueeze(1), labels



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

class whDataset(Dataset):
    def __init__(self, transform, num_samples=int(1e6), host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.transform = transform
        self.db_key = 'db1' # this is the append database, it should still work
        self.num_samples = num_samples
        self.host=host
        self.port=port
        r = redis.Redis(host=self.host, port=self.port)
        hang_until_redis_is_loaded(r)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        r = redis.Redis(host=self.host, port=self.port)
        index = index % r.llen(self.db_key)  # Use modular arithmetic to cycle through dataset
        pickled_data = r.lindex(self.db_key, index)
        if pickled_data is not None:
            data = pickle.loads(pickled_data)
            return self.transform(data[0]), self.transform(data[1])
        else:
            raise IndexError(f"Index {index} out of range")

# Create a dataset
dataset = whDataset(host=DEFAULT_HOST,port=DEFAULT_PORT, transform=rtransform,num_samples=WIREHEAD_NUMSAMPLES)
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=rcollate)

# Serve samples in batches from the DataLoader
for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
    # Here you can use the data and target for your model training
    pass

