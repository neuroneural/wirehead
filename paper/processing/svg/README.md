
---

# gpu

gpu_util : self explanatory
minute   : subsampled 60 second section of data

---

# experiment names

baseline             : naive generate -> train
wirehead_local       : generate -> wirehead -> train all on same node
wirehead_distributed : slurm -> generate -> wirehead -> train all on different nodes