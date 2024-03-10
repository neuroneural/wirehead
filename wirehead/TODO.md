# List of things that have to be working properly for wirehead mongo port

(*) means current task

- [ ] Generator
    - [x] Hook into Mongo
    - [x] Preprocessing  
    - [ ] Rewrite pushing function to use Sergey's schema
    - [ ] ID range assignment
    - [ ] Wrapper script for distributed deployment on slurm
    - [ ] Testing
- [ ] Manager
    - [x] Safe id iterator
        - [ ] (Optional) Figure out what to do in the unhappy code path 
    - [x] Safe singly get package function. __done, but this is somewhat slow due to exception handling__
    - [ ] Safe swap function
        - [ ] Make a different collection called 'ids' that contains only ids for valid packages
    - [ ] Testing
- [ ] Dataset
    - [ ] hopefully can reuse MongoDataset from mindfultensors
    - [ ] Testing

