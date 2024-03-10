# List of things that have to be working properly for wirehead mongo port

(*) means current task

- [ ] Generator
    - [x] Hook into Mongo
    - [x] Preprocessing  
    - [ ] Testing
- [ ] Manager
    - [x] Safe id iterator
        - [ ] (Optional) Figure out what to do in the unhappy code path 
    - [ ] Safe singly get package function (*)
    - [ ] Safe swap function
    - [ ] Testing
- [ ] Dataset
    - [ ] Hook into new manager, hopefully can reuse MongoDataset from mindfultensors
    - [ ] Testing

