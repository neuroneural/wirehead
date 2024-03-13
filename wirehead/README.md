### Structure ###

There are 3 core files to wirehead that should be run
- generator : manages the synthetic data generation pipeline, preprocessing and pushing to mongo
- manager   : manages the two part database (swapping and keeping stats)
- whdataset : custom pytorch dataset class to read from mongo
 
The other files contain utilities for the above files
- functions : contains networking, interfaces, preprocess and de-preprocessing functions
- defaults  : contains convenient defaults that are used globally in wirehead. some of these are custom to the arctic cluster used at trends

To reset the database safely, you should probably only do it manually:

