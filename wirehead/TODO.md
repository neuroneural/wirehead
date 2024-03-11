# List of things that have to be working properly for wirehead mongo port

(*) means current task

- [ ] Generator
    - [x] Hook into Mongo
    - [x] Preprocessing  
    - [x] Rewrite pushing function to use Sergey's schema
    - [ ] ID range assignment
    - [ ] Wrapper script for distributed deployment on slurm
    - [ ] Testing
- [ ] Manager
    - [x] Safe id iterator
        - [ ] (Optional) Figure out what to do in the unhappy code path 
    - [x] Safe singly get package function. __done, but this is somewhat slow due to exception handling__
    - [ ] Safe swap function
        - [x] create temp collection (because transactions aren't allowed now :/)
        - [x] remove all incomplete packages
        - [x] convert labels into contiguous collection.distinct -> sorted -> map (idx, order) 
        - [ ] rename to read
        - [ ] test
    - [ ] Testing
- [ ] Dataset
    - [ ] hopefully can reuse MongoDataset from mindfultensors
        - [ ] Read database must be contiguous indices [0..n]
    - [ ] Testing

