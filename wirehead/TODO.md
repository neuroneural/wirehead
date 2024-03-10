# List of things that have to be working properly for wirehead mongo port

(*) means current task

- [ ] Generator
    - [x] Hook into Mongo
    - [x] Preprocessing  
    - [ ] Testing
- [ ] Manager
    - [ ] Safe id iterator (*) __this is non trivial__
    - [ ] Safe singly get package function
    - [ ] Safe swap function
    - [ ] Testing
- [ ] Dataset
    - [ ] Hook into new manager, hopefully can reuse MongoDataset from mindfultensors
    - [ ] Testing

