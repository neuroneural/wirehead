

# samples read

convert to latex :<

---

# gpu

gpu_util : self explanatory

minute   : subsampled 60 second section of data, probably no smoothing

full     : full dataset, lots of smoothing

---

# experiment names

disk                 : naive load sample from disk -> train

baseline             : naive generate -> train

wirehead_local       : generate -> wirehead -> train all on same node

wirehead_distributed : slurm -> generate -> wirehead -> train all on different nodes
