SynthSeg Auto Redis Queue

* A project to make synthetic head generation using SynthSeg not take actual ages. *

---

Setup:

- conda activate wirehead
- launch redis on local node
- run generate.py on one terminal instance (this will just do it's business, pushing to the queue)
- when needed, launch dataloader.py to extract data out from the queue 

---

Optimizations on the table:

** Server side **

- faster GPUs for SynthSeg
- parallezie SynthSeg generation

** Client side **

- parallelize dataloader task using num_wokers
- 
