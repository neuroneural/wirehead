### From curriculum training script
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
    data = quantile_normalize(torch.from_numpy(batch[0][0]).float()).unsqueese(1)
    labels = torch.from_numpy(batch[0][1]).long().unsqueese(1)
    return data.unsqueeze(1), labels

class CustomRunner(dl.runner):
    ## Other stuff
    def get_loaders(self):
            # 'r'functions are just functions designed to work with wirehead
            
            rdataset = wh.whDataset(host=self.db_host,
                                       port=self.db_port,
                                       transform=rtransform,
                                       num_samples=WIREHEAD_NUMSAMPLES)

            rsampler = DistributedSampler(rdataset) if torch.cuda.device_count() > 1 else None

            tdataloader = BatchPrefetchLoaderWrapper(
                DataLoader(
                    rdataset,
                    sampler=rsampler,
                    collate_fn=rcollate,
                    pin_memory=True,
                    persistent_workers=True,
                    num_workers=1,
                ),
                num_prefetches=1,
            )
            return {"train": tdataloader}
    

class whDataset(Dataset):
    def __init__(self, transform, num_samples=int(1e6), host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.transform = transform
        self.db_key = 'db0' # change to 'db1' if this isn't working
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

