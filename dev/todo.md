# TODO

- Synthseg guide
    - slurm specifics

- Documentation
    - Internals: what manager does, what generator does
    - Deeper: what each function in either object does

- Technical
    - manager.py
        - swap() and other multi operation functions should have error handling
    - generate.py
        - push_and_insert() should assert that samples ["input", "label" ...] has same length as sample tuple
