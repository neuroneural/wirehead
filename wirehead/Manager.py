""" Wirehead Manager Class """

import time
import yaml
from pymongo import MongoClient, ASCENDING

class Manager():
    """ Manages the state of the mongo collections in Wirehead
    :param config_path  : path to yaml file containing wirehead configs
    """
    def __init__(self, config_path):
        # Loads variables from config file if path is specified
        if config_path is None:
            print("No config specified, exiting")
            return

        self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path):
        """ Loads manager configs from config_path """
        print("Manager: Config loaded from " + config_path)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        dbname = config.get('DBNAME')
        mongohost = config.get('MONGOHOST')
        client = MongoClient("mongodb://" + mongohost + ":27017")

        self.db         = client[dbname]
        self.swap_cap   = config.get('SWAP_CAP')
        self.sample     = tuple(config.get("SAMPLE"))
        self.COLLECTIONw = config.get("WRITE_COLLECTION") + ".bin"
        self.COLLECTIONr = config.get("READ_COLLECTION") + ".bin"
        self.COLLECTIONc = config.get("COUNTER_COLLECTION")
        self.COLLECTIONt = config.get("TEMP_COLLECTION") + ".bin"

    def run_manager(self):
        """ Initializes the database manager, swaps
            and cleans the database whenever swap_cap is hit """
        print("Manager: Initialized")
        self.db["status"].insert_one({"swapped": False})
        self.reset_counter_and_collection()
        generated = 0
        while True:
            generated = self.watch_and_swap(generated)

    def assert_sequence(self, collection):
        """Verify collection integrity"""
        unique_ids_count = len(collection.distinct("id"))
        assert (
            unique_ids_count == self.swap_cap
        ), f"Manager: Expected {self.swap_cap} unique ids, found {unique_ids_count}"
        expected_ids_set = set(range(self.swap_cap))
        actual_ids_set = set(collection.distinct("id"))
        assert (
            expected_ids_set == actual_ids_set
        ), "Manager: The ids aren't continuous from 0 to self.swap_cap - 1"

    def reset_counter_and_collection(self):
        """ Delete all documents in the main collection that have creeped in
            between the renaming and now. This operation is within a
            transaction """
        dbw = self.db[self.COLLECTIONw]
        dbc = self.db[self.COLLECTIONc]
        dbw.delete_many({}) # wipe the write collection
        # Reset the counter to zero
        _result = dbc.update_one(
            {"_id": "uniqueFieldCounter"},  # Query part: the document to match
            {
                "$set": {"sequence_value": 0}
            },  # Update part: what to set if the document is matched/found
            upsert=True,
        )
        dbw.delete_many({})
        dbw.create_index([("id", ASCENDING)], background=True)

    def swap(self, generated):
        """ Actions to be taken when the threshold is reached
            Renaming the collection and creating a new one """
        time.sleep(2) # Buffer for incomplete ops
        generated += self.swap_cap
        print("\n----swap----")
        print(f"Manager: Generated samples so far {generated}", flush=True)
        self.db[self.COLLECTIONw].rename(self.COLLECTIONt, dropTarget=True)
        # Now atomically reset the counter to 0 and delete whatever records
        # may have been written between the execution of the previous line
        # and the next
        self.reset_counter_and_collection()  # this is atomic
        result = self.db[self.COLLECTIONt].delete_many(
            {"id": {"$gt": self.swap_cap - 1}}
        )
        # Print the result of the deletion
        print(f"Manager: Documents deleted: {result.deleted_count}", flush=True)
        self.assert_sequence(self.db[self.COLLECTIONt])
        self.db[self.COLLECTIONt].rename(self.COLLECTIONr, dropTarget=True)
        self.db["status"].insert_one({"swapped": True})
        return generated

    def watch_and_swap(self, generated):
        """ Watch the write collection and swap when full"""
        counter_doc = self.db[self.COLLECTIONc].find_one(
            {"_id": "uniqueFieldCounter"})
        if counter_doc["sequence_value"] >= self.swap_cap:  # watch
            return self.swap(generated)                     # swap
        return generated