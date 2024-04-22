# wirehead #

Caching system for horizontal scaling of synthetic data generators using MongoDB

---

## Usage ## 

See examples/synthseg/worker.py for a detailed example 

```
from wirehead import Runtime 
from pymongo import MongoClient

# Mongo config
DBNAME              = "wirehead_mike"
MONGOHOST           = "arctrdcn018.rs.gsu.edu"
client              = MongoClient("mongodb://" + MONGOHOST + ":27017")
db                  = client[DBNAME]

# Declare wirehead runtime object

def create_generate():
    """ Yields a tuple of type ((a, b,...), ('kind_a', 'kind_b',...))"""
    ...
    yield ret 

generator           = create_generator(my_task_id())
wirehead_runtime    = Runtime(
    db = db,                    # Specify mongohost
    generator = generator,      # Specify generator 
)
```

Then, to run the generator, simply do 

```
wirehead_runtime.run_generator()
```

Or, to run the database manager,

```
wirehead_runtime.run_manager()
```
