class Runtime():
    """ Wirehead runtime class, which wraps around the generator
        and manager runtimes."""
    def __init__(self,
                 db, 
                 generator, 
                 cap=1000, 
                 wcount=1, 
                 log_metrics=False):
        self.db                 = db
        self.generator          = generator 
        self.swap_cap           = cap
        self.COLLECTIONw        = "write.bin"
        self.COLLECTIONr        = "read.bin"
        self.COLLECTIONt        = "temp.bin"
        self.COLLECTIONc        = "counters"
        self.COLLECTIONm        = 'metrics'
        self.CHUNKSIZE          = 10
        self.LOG_METRICS        = True
        self.EXPERIMENT_KIND    = ''
        self.EXPERIMENT_NAME    = ''
        self.WORKER_COUNT            = wcount
        self.LOCAL_RUNNING           = True
    
    # Manager Ops
    def assert_sequence(self, collection):
        """Verify collection integrity"""
        unique_ids_count = len(collection.distinct("id"))
        assert (
            unique_ids_count == self.swap_cap
        ), f"Expected {self.swap_cap} unique ids, found {unique_ids_count}"
        expected_ids_set = set(range(self.swap_cap))
        actual_ids_set = set(collection.distinct("id"))
        assert (
            expected_ids_set == actual_ids_set
        ), "The ids aren't continuous from 0 to self.swap_cap - 1"

    def reset_counter_and_collection(write_collection, counter_collection):
        """ Delete all documents in the main collection that have creeped in
            between the renaming and now. This operation is within a
            transaction """
        write_collection.delete_many({})
        # Reset the counter to zero
        result = counter_collection.update_one(
            {"_id": "uniqueFieldCounter"},  # Query part: the document to match
            {
                "$set": {"sequence_value": 0}
            },  # Update part: what to set if the document is matched/found
            upsert=True,
        )
        # Delete all documents in the main collection that have creeped in
        # between the renaming and now. This operation is within a
        # transaction
        write_collection.delete_many({})
        write_collection.create_index([("id", ASCENDING)], background=True)

    def log_metrics(self, generated):
        """ Inserts run metrics into COLLECTIONm """
        metrics_collection  = self.db[COLLECTIONm]
        # Create a BSON document with the metrics
        metrics_doc = {
            "experiment_name": self.EXPERIMENT_NAME,
            "kind": self.EXPERIMENT_KIND,
            "timestamp": time.time(),
            "total_samples": generated,
            "worker_count": self.WORKER_COUNT.
        }
        # Insert the metrics document into the metrics collection
        print('Manager: Swap metrics logged')
        metrics_collection.insert_one(metrics_doc)

    def watch_and_swap(self,generated):
        """ Watch the write collection and swap when full"""
        def swap(self, generated):
            """ Actions to be taken when the threshold is reached
                Renaming the collection and creating a new one """
            time.sleep(2)
            generated += self.swap_cap
            print(f"Generated samples so far {generated}", flush=True)
            self.db[self.COLLECTIONw].rename(self.COLLECTIONt, dropTarget=True)
            # Now atomically reset the counter to 0 and delete whatever records
            # may have been written between the execution of the previous line
            # and the next
            reset_counter_and_collection(
                self.db[self.COLLECTIONw], self.db[self.COLLECTIONc]
            )  # this is atomic
            result = self.db[self.COLLECTIONt].delete_many(
                {"id": {"$gt": self.swap_cap - 1}}
            )
            # Print the result of the deletion
            print(f"Documents deleted: {result.deleted_count}", flush=True)
            self.assert_sequence(self.db[self.COLLECTIONt], self.swap_cap)
            self.db[COLLECTIONt].rename(self.COLLECTIONr, dropTarget=True)

            if self.LOG_METRICS: log_metrics(generated)
            return generated

        counter_doc = self.db[self.COLLECTIONc].find_one(
            {"_id": "uniqueFieldCounter"})
        if counter_doc["sequence_value"] >= self.swap_cap:  # watch
            return swap(generated)                          # swap
        return generated

    # Generator Ops
    def tensor2bin(tensor):
        """Seralize a torch tensor into an IO buffer"""
        tensor_1d = tensor.to(torch.uint8)
        buffer = io.BytesIO()
        torch.save(tensor_1d, buffer)
        tensor_binary = buffer.getvalue()
        return tensor_binary

    def generate_and_insert(collection_bin, counter_collection, chunk_size):
        """ Fetch from generator and inserts into mongodb """
        # 0. Fetch data from generator
        data = next(self.generator)
        # 1. Get the correct index for this current sample
        index = get_current_idx(counter_collection)
        # 2. Turn the data into a list of serialized chunks  
        chunks = chunkify(data, index, chunk_size)
        # 3. Push to mongodb + error handling
        push_chunks(collection_bin, chunks)

    

    



    

def my_task_id() -> int:
    task_id = os.getenv(
        "SLURM_ARRAY_TASK_ID", "0")
    return int(task_id)

    

    







