""" Wirehead Manager Class """

import os
import time
import yaml
from pymongo import MongoClient, ASCENDING


class WireheadManager:
    """
    Manages the state of the mongo collections in Wirehead.

    :param config_path: path to yaml file containing wirehead configs
    """

    def __init__(self, config_path):
        if config_path is None or os.path.exists(config_path) is False:
            print("No valid config specified, exiting")
            return
        self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path):
        """
        Loads manager configs from config_path.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        dbname = config.get("DBNAME")
        mongohost = config.get("MONGOHOST")
        port = config.get("PORT") if config.get("PORT") is not None else 27017
        client = MongoClient("mongodb://" + mongohost + ":" + str(port))

        self.db = client[dbname]
        self.swap_cap = config.get("SWAP_CAP")
        self.collectionw = config.get("WRITE_COLLECTION") + ".bin"
        self.collectionr = config.get("READ_COLLECTION") + ".bin"
        self.collectionc = config.get("COUNTER_COLLECTION")
        self.collectiont = config.get("TEMP_COLLECTION") + ".bin"
        self.expected_ids_set = set(range(self.swap_cap))

    def verify_collection_integrity(self, collection):
        """
        Verifies collection contains contiguous elements with id 0..swap_cap
        """
        unique_ids_count = len(collection.distinct("id"))
        if unique_ids_count != self.swap_cap:
            print(
                f"Manager: Expected {self.swap_cap} unique ids, found {unique_ids_count}"
            )
            return False
        # can be factored to make faster (use numpy?)
        actual_ids_set = set(collection.distinct("id"))
        if self.expected_ids_set != actual_ids_set:
            print(
                "Manager: The ids aren't continuous from 0 to self.swap_cap - 1"
            )
            return False
        # If all checks pass
        return True

    def reset_counter_and_collection(self):
        """
        Delete all documents in the main collection that have creeped in
        between the renaming and now. This operation is within a transaction.
        """
        dbw = self.db[self.collectionw]
        dbc = self.db[self.collectionc]
        dbw.delete_many({})  # wipe the write collection
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
        """
        Moves data from write collection to read collection
        Deletes old write collection
        Maintains data integrity in between
        """
        time.sleep(2)  # Buffer for incomplete ops
        generated += self.swap_cap
        print(
            f"Manager: Time: {time.time()} Generated samples so far {generated}"
        )
        self.db[self.collectionw].rename(self.collectiont, dropTarget=True)
        # Now atomically reset the counter to 0 and delete whatever records
        # may have been written between the execution of the previous line
        # and the next
        self.reset_counter_and_collection()  # this is atomic
        result = self.db[self.collectiont].delete_many(
            {"id": {"$gt": self.swap_cap - 1}}
        )
        # Print the result of the deletion
        print(f"Manager: Documents deleted: {result.deleted_count}")
        if self.verify_collection_integrity(self.db[self.collectiont]):
            self.db[self.collectiont].rename(self.collectionr, dropTarget=True)
            self.db["status"].insert_one({"swapped": True})
            return generated
        else:
            print("Manager: Corrupted collection detected, skipping swap")
            return generated

    def watch_and_swap(self, generated):
        """
        Watch the write collection and swap when full
        """
        counter_doc = self.db[self.collectionc].find_one(
            {"_id": "uniqueFieldCounter"}
        )
        if counter_doc["sequence_value"] > self.swap_cap:  # watch
            return self.swap(generated)  # swap
        return generated

    def run_manager(self):
        """
        Initializes the database manager, swaps and cleans the database whenever swap_cap is hit.
        """
        print("Manager: Initialized")
        self.db["status"].insert_one({"swapped": False})
        self.reset_counter_and_collection()
        generated = 0
        while True:
            generated = self.watch_and_swap(generated)
